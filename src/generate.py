"""
Генерация примеров редких классов (privilege, exfiltration) через Groq API.
Модель: llama-3.3-70b-versatile (бесплатно, 14 400 запросов/день).
Результат сохраняется в data/processed/generated.csv.

Использование:
    python src/generate.py --n 60
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"

PROMPTS = {
    "privilege": """Generate {n} different SQL queries used for privilege escalation in databases. Include:
- GRANT ALL PRIVILEGES / creating admin users
- xp_cmdshell (MSSQL), UTL_FILE (Oracle)
- access to sys, information_schema, pg_catalog system tables
- ALTER USER, CREATE ROLE with broad permissions

Return ONLY a JSON array of strings, each string is one SQL query. No explanations.
Example: ["GRANT ALL ON *.* TO 'x'@'%';", "EXEC xp_cmdshell 'whoami';"]""",

    "exfiltration": """Generate {n} SQL queries for hidden data exfiltration. Include:
- SELECT with UNION to read system tables
- queries targeting password, token, PII tables
- INTO OUTFILE (MySQL), COPY TO (PostgreSQL)
- xp_dirtree, UTL_HTTP for out-of-band exfiltration
- broad SELECT without filters (SELECT * FROM users)

Return ONLY a JSON array of strings. No explanations.""",

    "legit": """Generate {n} legitimate SQL queries for a typical web application. Include:
- SELECT with JOIN, WHERE, ORDER BY, GROUP BY
- INSERT, UPDATE, DELETE for business logic
- parameterized queries (use ? or $1 as placeholders)
- CREATE TABLE, ALTER TABLE for migrations

Return ONLY a JSON array of strings. No explanations.""",
}


BATCH_SIZE = 60  # максимум за один запрос


def _parse_json(raw: str) -> list[str]:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # обрезаем незакрытый массив если модель не уложилась в токены
    if not raw.endswith("]"):
        last_comma = raw.rfind('",')
        if last_comma != -1:
            raw = raw[: last_comma + 1] + "]"
    return json.loads(raw)


def _generate_batch(label_name: str, n: int) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPTS[label_name].format(n=n)}],
        max_tokens=8192,
        temperature=0.9,
    )
    return _parse_json(response.choices[0].message.content)


def generate_class(label_name: str, label_id: int, n: int) -> list[dict]:
    print(f"[*] Generating '{label_name}' ({n} examples in batches of {BATCH_SIZE})...")
    queries: list[str] = []

    while len(queries) < n:
        need = min(BATCH_SIZE, n - len(queries))
        try:
            batch = _generate_batch(label_name, need)
            queries.extend(batch)
            print(f"    batch done: {len(batch)} -> total {len(queries)}/{n}")
        except Exception as e:
            print(f"    [!] batch failed: {e}, retrying...")
        time.sleep(1)

    return [
        {"query": q, "label": label_id, "label_name": label_name, "source": "generated_groq"}
        for q in queries[:n]
    ]


def main(n: int = 60) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []

    for label_name, label_id in [("privilege", 3), ("exfiltration", 4), ("legit", 0)]:
        count = n if label_name != "legit" else n // 2
        rows.extend(generate_class(label_name, label_id, count))

    df = pd.DataFrame(rows)
    out = DATA_PROCESSED / "generated.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[+] Saved {len(df)} examples -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Examples per class")
    args = parser.parse_args()
    main(n=args.n)
