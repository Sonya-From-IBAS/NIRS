"""
Версии исследовательских данных в PostgreSQL и исторические CSV-утилиты.

Новые выборки загружаются через load_source_reference/load_verified.
CSV-сборка и её старые эвристики сохранены для исторического контроля;
по умолчанию CLI не запускает их и не перезаписывает dataset.csv.

Классы угроз (label):
    0 - legit          легитимный SQL
    1 - sqli           SQL-инъекции (classic, blind, time-based)
    2 - obfuscated     обфусцированный SQL (hex, CHAR, комментарии)
    3 - privilege      эскалация привилегий (xp_cmdshell, GRANT, sys.*)
    4 - exfiltration   скрытая эксфильтрация данных
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

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


def source_stratified_split(df: pd.DataFrame,
                            test_size: float = 0.2,
                            random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Стратифицированный сплит по (label × source).
    Гарантирует, что синтетические примеры каждого источника
    представлены и в train, и в test — исключает ситуацию,
    когда весь generated_groq_mssql_advanced попадает только в train.
    """
    # Создаём составной ключ стратификации
    df = df.copy()
    # Нормализуем source: группируем редкие источники
    source_norm = df["source"].fillna("unknown")
    # Для источников с < 10 примерами используем только label как стратификацию
    source_counts = source_norm.value_counts()
    rare = source_counts[source_counts < 10].index
    source_norm = source_norm.apply(lambda s: "other_" + str(df.loc[source_norm == s, "label"].iloc[0])
                                    if s in rare else s)
    df["_strat_key"] = df["label"].astype(str) + "__" + source_norm

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["_strat_key"],
    )
    train_df = df.loc[train_idx].drop(columns=["_strat_key"])
    test_df  = df.loc[test_idx].drop(columns=["_strat_key"])

    print(f"Split: train={len(train_df)}, test={len(test_df)}")
    print("Source distribution in test:")
    print(test_df.groupby(["label_name", "source"]).size()
          .reset_index(name="count").to_string(index=False))
    return train_df, test_df


def print_stats(df: pd.DataFrame) -> None:
    print("\n=== Dataset stats ===")
    print(f"Total: {len(df)} examples\n")
    stats = df.groupby(["label", "label_name"]).size().reset_index(name="count")
    stats["pct"] = (stats["count"] / len(df) * 100).round(1)
    print(stats.to_string(index=False))
    if "source" in df.columns:
        print("\nBy source:")
        print(df.groupby("source").size().sort_values(ascending=False).to_string())
    print()


def load_source_reference(dataset_version_id: str, store=None) -> pd.DataFrame:
    """Five-class original-source reference, explicitly not verified binary truth."""
    if store is None:
        from storage import Storage
        store = Storage()
    with store.connect() as connection:
        rows = connection.execute("""SELECT d.ordinal AS row_id,o.observation_id,o.sql_text AS query,
            c.source_name AS source,c.source_label
            FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE d.dataset_version_id=%s ORDER BY d.ordinal""", (dataset_version_id,)).fetchall()
    values = []
    for row in rows:
        source_label = row.pop("source_label")
        label = source_label.get("label") if isinstance(source_label, dict) else None
        if str(label) not in {str(key) for key in LABEL_NAMES}:
            raise ValueError("This source does not provide the historical five-class label space")
        row.update(label=int(label), label_name=LABEL_NAMES[int(label)], label_basis="original_source_reference")
        values.append(row)
    return pd.DataFrame(values, columns=["row_id", "observation_id", "query", "source", "label", "label_name", "label_basis"])


def load_verified(dataset_version_id: str, store=None) -> pd.DataFrame:
    """Binary targets pinned by this dataset version; service fields stay separate."""
    if store is None:
        from storage import Storage
        store = Storage()
    with store.connect() as connection:
        rows = connection.execute("""SELECT d.ordinal AS row_id,o.observation_id,o.sql_text AS query,
            a.target_label AS label,a.data->'threat_tags' AS threat_tags,c.source_name AS source
            FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
            JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
            JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
            WHERE d.dataset_version_id=%s AND a.status='verified' AND o.unit_kind='full_sql' AND o.purpose='research'
            ORDER BY d.ordinal""", (dataset_version_id,)).fetchall()
    return pd.DataFrame(rows, columns=["row_id", "observation_id", "query", "label", "threat_tags", "source"])


if __name__ == "__main__":
    import argparse
    import json
    from storage import Storage
    parser = argparse.ArgumentParser(description="Read versioned PostgreSQL datasets; original CSVs are not overwritten.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for mode in ("export-reference", "export-verified"):
        command = commands.add_parser(mode)
        command.add_argument("dataset_version_id")
        command.add_argument("output", type=Path)
    args = parser.parse_args()
    storage = Storage()
    if args.command == "list":
        with storage.connect() as connection:
            rows = connection.execute("""SELECT v.dataset_version_id,v.name,v.status,count(d.observation_id) AS rows
                FROM nirs.dataset_versions v LEFT JOIN nirs.dataset_members d USING(dataset_version_id)
                GROUP BY v.dataset_version_id ORDER BY v.name""").fetchall()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        frame = (load_source_reference if args.command == "export-reference" else load_verified)(args.dataset_version_id, storage)
        protected = {path.resolve() for path in (DATA_PROCESSED / "dataset.csv", DATA_PROCESSED / "generated.csv")}
        if args.output.resolve() in protected:
            raise ValueError("Export must not overwrite original inputs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Exported {len(frame)} rows")
