"""
Загрузка реальных SQL-пейлоадов из публичных репозиториев GitHub.
Источники:
  - PayloadsAllTheThings (swisskyrepo)
  - Advanced-SQL-Injection-Cheatsheet (kleiton0x00)
  - SQLmap tamper scripts (obfuscation)
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

DATA_RAW       = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}

PATT_BASE = "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/SQL%20Injection"

SOURCES = {
    # privilege escalation
    3: [
        f"{PATT_BASE}/MSSQL%20Injection.md",
        f"{PATT_BASE}/MySQL%20Injection.md",
        f"{PATT_BASE}/PostgreSQL%20Injection.md",
        f"{PATT_BASE}/OracleSQL%20Injection.md",
    ],
    # exfiltration
    4: [
        f"{PATT_BASE}/MSSQL%20Injection.md",
        f"{PATT_BASE}/MySQL%20Injection.md",
        f"{PATT_BASE}/PostgreSQL%20Injection.md",
        f"{PATT_BASE}/OracleSQL%20Injection.md",
    ],
    # obfuscated
    2: [
        f"{PATT_BASE}/MySQL%20Injection.md",
        f"{PATT_BASE}/MSSQL%20Injection.md",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQLi/quick-SQLi.txt",
    ],
}

LABEL_NAMES = {2: "obfuscated", 3: "privilege", 4: "exfiltration"}

# ── Ключевые слова для фильтрации из markdown ─────────────────────────────

PRIVILEGE_KEYWORDS = re.compile(
    r"\b(GRANT|REVOKE|xp_cmdshell|xp_dirtree|UTL_FILE|UTL_HTTP|CREATE\s+USER"
    r"|ALTER\s+USER|CREATE\s+ROLE|sp_addsrvrolemember|sys\.\w+|pg_shadow"
    r"|mysql\.user|EXEC\s+master|openrowset|bulk\s+insert)\b",
    re.IGNORECASE,
)

EXFIL_KEYWORDS = re.compile(
    r"\b(INTO\s+OUTFILE|INTO\s+DUMPFILE|LOAD_FILE|COPY\s+TO|UTL_HTTP\s*\."
    r"|xp_dirtree|dns|out.of.band|base64|hex\(|TO_BASE64|UTL_INADDR)\b",
    re.IGNORECASE,
)

OBFUSC_KEYWORDS = re.compile(
    r"(0x[0-9a-fA-F]{4,}|CHAR\s*\(\s*\d|/\*.*?\*/|%[0-9a-fA-F]{2}"
    r"|CONCAT\s*\(CHAR|\|\|CHAR|CHR\s*\()",
    re.IGNORECASE,
)

LABEL_FILTERS = {
    3: PRIVILEGE_KEYWORDS,
    4: EXFIL_KEYWORDS,
    2: OBFUSC_KEYWORDS,
}


def fetch(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"    [!] Failed {url}: {e}")
        return ""


def extract_sql_from_markdown(text: str) -> list[str]:
    """Вытаскивает строки из code-блоков markdown."""
    queries = []
    # блоки ```sql ... ``` и ``` ... ```
    blocks = re.findall(r"```(?:sql|SQL)?\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if len(line) > 10 and not line.startswith("--") and not line.startswith("#"):
                queries.append(line)
    # отдельные строки, начинающиеся с SQL-ключевых слов
    sql_start = re.compile(
        r"^(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|GRANT|EXEC|EXECUTE"
        r"|UNION|WITH|TRUNCATE|CALL|xp_|sp_|UTL_|COPY\s)",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        line = line.strip().lstrip("`").rstrip("`")
        if sql_start.match(line) and len(line) > 10:
            queries.append(line)
    return list(set(queries))


def main() -> None:
    rows = []

    for label_id, urls in SOURCES.items():
        label_name = LABEL_NAMES[label_id]
        keyword_filter = LABEL_FILTERS[label_id]
        collected = []

        for url in urls:
            print(f"[*] Fetching {url[:70]}...")
            text = fetch(url)
            if not text:
                continue
            candidates = extract_sql_from_markdown(text)
            matched = [q for q in candidates if keyword_filter.search(q)]
            collected.extend(matched)
            print(f"    found {len(matched)} matching queries")
            time.sleep(1)

        collected = list(set(collected))
        print(f"[+] {label_name}: {len(collected)} unique real examples\n")

        for q in collected:
            rows.append({
                "query": q,
                "label": label_id,
                "label_name": label_name,
                "source": "real_github",
            })

    if not rows:
        print("[!] Nothing collected. Check network or URLs.")
        return

    df = pd.DataFrame(rows)
    out = DATA_RAW / "real_payloads.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[+] Saved {len(df)} real examples -> {out}")
    print(df["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
