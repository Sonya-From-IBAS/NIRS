"""
Загрузка, нормализация и разметка датасетов по вредоносному SQL.

Классы угроз (label):
    0 - legit          легитимный SQL
    1 - sqli           SQL-инъекции (classic, blind, time-based)
    2 - obfuscated     обфусцированный SQL (hex, CHAR, комментарии)
    3 - privilege      эскалация привилегий (xp_cmdshell, GRANT, sys.*)
    4 - exfiltration   скрытая эксфильтрация данных
"""

import re
import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

LABEL_NAMES = {
    0: "legit",
    1: "sqli",
    2: "obfuscated",
    3: "privilege",
    4: "exfiltration",
}

# ── Kaggle: sql-injection-dataset ──────────────────────────────────────────
# Ожидаемый файл: data/raw/sqli_dataset.csv
# Колонки: Sentence, Label  (Label: 0=legit, 1=sqli)

def load_kaggle_sqli(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "sqli_dataset.csv"
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # приводим к общей схеме
    df = df.rename(columns={"Query": "query", "Sentence": "query", "Label": "label"})
    df = df[["query", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["source"] = "kaggle_sqli"
    return df


# ── SecLists: SQL-инъекции и обфускация ────────────────────────────────────
# Источник: https://github.com/danielmiessler/SecLists
# Файлы в data/raw/seclists_sqli.txt  (один payload на строку)

def load_seclists(path: Path | None = None, label: int = 1) -> pd.DataFrame:
    path = path or DATA_RAW / "seclists_sqli.txt"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    queries = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    df = pd.DataFrame({"query": queries})
    df["label"] = label
    df["source"] = "seclists"
    return df


# ── Генерированные примеры (из generate.py) ────────────────────────────────
def load_generated(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_PROCESSED / "generated.csv"
    return pd.read_csv(path, encoding="utf-8")


# ── Определение обфускации по строгой эвристике ───────────────────────────
# Используем БОЛЕЕ СТРОГИЕ паттерны чем в RegexDetector,
# чтобы избежать утечки данных: требуем 2+ признака или явное кодирование.
_STRONG_HEX   = re.compile(r"0x[0-9a-fA-F]{6,}", re.IGNORECASE)           # длинный hex
_CHAR_SEQ     = re.compile(r"CHAR\s*\(\s*\d+\s*\)", re.IGNORECASE)        # CHAR()
_INLINE_CMT   = re.compile(r"/\*[^*]{0,20}\*/", re.IGNORECASE)            # /**/
_URL_ENCODE   = re.compile(r"%[0-9a-fA-F]{2}.*%[0-9a-fA-F]{2}")           # %xx%xx
_CONCAT_CHAR  = re.compile(r"(CONCAT|CHR|CHAR)\s*\(.+\|\|", re.IGNORECASE)


def _obfusc_score(query: str) -> int:
    """Считает сколько сигналов обфускации присутствует."""
    score = 0
    if _STRONG_HEX.search(query):   score += 2  # hex сам по себе достаточен
    if _CHAR_SEQ.search(query):      score += 2
    if _INLINE_CMT.search(query):    score += 1
    if _URL_ENCODE.search(query):    score += 1
    if _CONCAT_CHAR.search(query):   score += 1
    return score


def reclassify_obfuscated(df: pd.DataFrame) -> pd.DataFrame:
    """Помечает sqli-примеры как класс 2 только при score >= 2."""
    mask = (df["label"] == 1) & (df["query"].apply(_obfusc_score) >= 2)
    df.loc[mask, "label"] = 2
    return df


# ── Реальные пейлоады из GitHub ────────────────────────────────────────────
def load_real_payloads(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "real_payloads.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


# ── Сборка финального датасета ─────────────────────────────────────────────
def build_dataset(save: bool = True, filename: str = "dataset.csv") -> pd.DataFrame:
    frames = []

    kaggle_path = DATA_RAW / "sqli_dataset.csv"
    if kaggle_path.exists():
        frames.append(load_kaggle_sqli(kaggle_path))
        print(f"[+] Kaggle SQLi: {len(frames[-1])} rows")

    seclists_path = DATA_RAW / "seclists_sqli.txt"
    if seclists_path.exists():
        frames.append(load_seclists(seclists_path, label=1))
        print(f"[+] SecLists: {len(frames[-1])} rows")

    generated_path = DATA_PROCESSED / "generated.csv"
    if generated_path.exists():
        frames.append(load_generated(generated_path))
        print(f"[+] Generated: {len(frames[-1])} rows")

    real_path = DATA_RAW / "real_payloads.csv"
    real_df = load_real_payloads(real_path)
    if not real_df.empty:
        frames.append(real_df)
        print(f"[+] Real payloads (GitHub): {len(real_df)} rows")

    if not frames:
        raise FileNotFoundError("No datasets found in data/raw/")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="query")
    df = reclassify_obfuscated(df)
    df["label_name"] = df["label"].map(LABEL_NAMES)

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out = DATA_PROCESSED / filename
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[+] Saved: {out} ({len(df)} rows)")

    return df


def print_stats(df: pd.DataFrame) -> None:
    print("\n=== Статистика датасета ===")
    print(f"Всего: {len(df)} примеров\n")
    stats = df.groupby(["label", "label_name"]).size().reset_index(name="count")
    stats["pct"] = (stats["count"] / len(df) * 100).round(1)
    print(stats.to_string(index=False))
    print()


if __name__ == "__main__":
    df = build_dataset()
    print_stats(df)
