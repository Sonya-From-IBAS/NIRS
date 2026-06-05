"""
Генерация синтетических примеров через Groq API.
Улучшения v2:
  - Генерация раздельно по СУБД (MySQL, MSSQL, PostgreSQL, Oracle)
  - Разные уровни сложности (простой / продвинутый)
  - Дедупликация по similarity перед сохранением
  - Тег source содержит СУБД и сложность для source-стратификации сплита

Использование:
    python src/generate.py --n 80
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"  # высокий rate limit (20k токенов/мин)
BATCH_SIZE = 20

# ── Шаблоны промптов по СУБД и классу ─────────────────────────────────────

PRIVILEGE_PROMPTS = {
    "mysql":      "Generate {n} SQL privilege escalation queries for MySQL. Use: GRANT ALL, CREATE USER, mysql.user access, FILE privilege, INTO OUTFILE for RCE, UDF loading.",
    "mssql":      "Generate {n} SQL privilege escalation queries for Microsoft SQL Server. Use: xp_cmdshell, sp_addsrvrolemember, EXEC master.., openrowset, linked servers, impersonation with EXECUTE AS.",
    "postgresql": "Generate {n} SQL privilege escalation queries for PostgreSQL. Use: COPY TO/FROM PROGRAM, pg_read_file, CREATE EXTENSION, pg_shadow access, ALTER ROLE SUPERUSER, lo_export.",
    "oracle":     "Generate {n} SQL privilege escalation queries for Oracle DB. Use: UTL_FILE, UTL_HTTP, DBMS_SCHEDULER, CREATE ANY PROCEDURE, GRANT DBA, SYS.ALL_TABLES access.",
}

EXFIL_PROMPTS = {
    "mysql":      "Generate {n} SQL data exfiltration queries for MySQL. Use: SELECT INTO OUTFILE, LOAD_FILE, GROUP_CONCAT to aggregate data, queries on mysql.user/information_schema, UNION-based data dump.",
    "mssql":      "Generate {n} SQL data exfiltration queries for MSSQL. Use: xp_dirtree for DNS exfil, OPENROWSET to send data out, FOR XML PATH to concatenate rows, linked server abuse, bulk export.",
    "postgresql": "Generate {n} SQL data exfiltration queries for PostgreSQL. Use: COPY table TO '/tmp/out', pg_read_file, dblink for out-of-band, UNION SELECT from pg_shadow, large SELECT * without WHERE.",
    "oracle":     "Generate {n} SQL data exfiltration queries for Oracle. Use: UTL_HTTP.request to external host, UTL_FILE.fopen, SYS.DBMS_ADVISOR, SELECT * FROM ALL_USERS, XMLType for data encoding.",
}

LEGIT_PROMPTS = {
    "mysql":      "Generate {n} legitimate MySQL queries for a web app (e-commerce or CRM). Use real table names, JOINs, indexes, parameterized with ?.",
    "mssql":      "Generate {n} legitimate T-SQL queries for a web app. Use TOP, WITH (NOLOCK), stored procedures, parameterized with @param.",
    "postgresql": "Generate {n} legitimate PostgreSQL queries for a web app. Use CTEs, RETURNING, parameterized with $1, JSONB fields.",
    "oracle":     "Generate {n} legitimate Oracle SQL queries for a web app. Use ROWNUM, NVL, TO_DATE, DECODE, parameterized with :param.",
}

COMPLEXITY = ["simple", "advanced"]

COMPLEXITY_SUFFIX = {
    "simple":   " Keep queries short (1-2 lines). Avoid chaining multiple statements.",
    "advanced": " Make queries complex: subqueries, CTEs, multi-step attacks, evasion techniques.",
}

LABEL_PROMPTS = {
    "privilege":   PRIVILEGE_PROMPTS,
    "exfiltration": EXFIL_PROMPTS,
    "legit":       LEGIT_PROMPTS,
}

LABEL_IDS = {"privilege": 3, "exfiltration": 4, "legit": 0}

RETURN_INSTRUCTION = "\n\nReturn ONLY a JSON array of strings, one SQL query per element. No explanations, no markdown."


# ── Парсинг ответа ─────────────────────────────────────────────────────────

def _parse_json(raw: str) -> list[str]:
    # убираем markdown-обёртку
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # пробуем найти JSON-массив в тексте
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    # если массив обрезан — закрываем его
    if not raw.endswith("]"):
        last = raw.rfind('",')
        if last != -1:
            raw = raw[:last + 1] + "]"

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # фолбэк: вытаскиваем строки между кавычками напрямую
        result = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)

    # flatten если вдруг вернулся список списков
    flat = []
    for item in result:
        if isinstance(item, list):
            flat.extend(str(x) for x in item)
        elif isinstance(item, str) and len(item) > 3:
            flat.append(item)
    return flat


def _generate_batch(prompt: str, n: int) -> list[str]:
    full_prompt = prompt + RETURN_INSTRUCTION
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt.replace("{n}", str(n))}],
        max_tokens=2000,  # 20 запросов × ~50 токенов = ~1000, берём с запасом
        temperature=0.95,
    )
    return _parse_json(response.choices[0].message.content)


# ── Дедупликация ───────────────────────────────────────────────────────────

def _deduplicate(queries: list[str], threshold: float = 0.85) -> list[str]:
    """Убирает почти-дубли по символьному overlap (простой Jaccard на trigrams)."""
    def trigrams(s: str) -> set:
        s = s.lower().strip()
        return {s[i:i+3] for i in range(len(s) - 2)}

    unique = []
    seen_trigrams = []
    for q in queries:
        tg = trigrams(q)
        if not tg:
            continue
        is_dup = any(
            len(tg & prev) / max(len(tg | prev), 1) > threshold
            for prev in seen_trigrams
        )
        if not is_dup:
            unique.append(q)
            seen_trigrams.append(tg)
    return unique


# ── Генерация одного класса ────────────────────────────────────────────────

MAX_FAILS = 3  # максимум провалов подряд перед пропуском варианта


def generate_class(label_name: str, n_per_variant: int, out_path: Path) -> list[dict]:
    label_id = LABEL_IDS[label_name]
    prompts_by_db = LABEL_PROMPTS[label_name]
    all_rows = []

    for db, prompt_template in prompts_by_db.items():
        for complexity in COMPLEXITY:
            prompt = prompt_template + COMPLEXITY_SUFFIX[complexity]
            source_tag = f"generated_groq_{db}_{complexity}"
            collected: list[str] = []
            fails = 0

            print(f"  [{label_name}] db={db} complexity={complexity} target={n_per_variant}", flush=True)

            while len(collected) < n_per_variant and fails < MAX_FAILS:
                need = min(BATCH_SIZE, n_per_variant - len(collected))
                try:
                    batch = _generate_batch(prompt, need)
                    collected.extend(batch)
                    fails = 0
                    print(f"      batch: {len(batch)} -> {len(collected)}/{n_per_variant}", flush=True)
                except Exception as e:
                    fails += 1
                    print(f"      [!] failed ({fails}/{MAX_FAILS}): {e}", flush=True)
                time.sleep(1.5)

            if fails >= MAX_FAILS:
                print(f"      [!] skipped after {MAX_FAILS} consecutive failures", flush=True)

            collected = _deduplicate(collected[:n_per_variant])
            for q in collected:
                all_rows.append({
                    "query": q,
                    "label": label_id,
                    "label_name": label_name,
                    "source": source_tag,
                })

            # инкрементальное сохранение после каждого варианта
            if all_rows:
                pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8")

    print(f"[+] {label_name}: {len(all_rows)} unique examples", flush=True)
    return all_rows


# ── Main ───────────────────────────────────────────────────────────────────

def main(n_per_variant: int = 20) -> None:
    """
    n_per_variant: примеров на каждую (СУБД × сложность) комбинацию.
    Итого на класс: n_per_variant × 4 СУБД × 2 уровня = n_per_variant × 8.
    При n=20: 160 примеров на класс, итого ~480.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []

    out = DATA_PROCESSED / "generated.csv"
    for label_name in ["privilege", "exfiltration", "legit"]:
        n = n_per_variant if label_name != "legit" else n_per_variant // 2
        rows.extend(generate_class(label_name, n, out))

    df = pd.DataFrame(rows)
    out = DATA_PROCESSED / "generated.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[+] Saved {len(df)} examples -> {out}")
    print(df.groupby(["label_name", "source"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Примеров на каждую (СУБД x сложность) комбинацию")
    args = parser.parse_args()
    main(n_per_variant=args.n)
