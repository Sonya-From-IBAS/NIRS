"""
Загрузка публичных датасетов.

Датасет 1 (Kaggle):
    kaggle datasets download -d sajid576/sql-injection-dataset
    Файл: sqli_dataset.csv  (Label: 0=legit, 1=sqli, ~34k строк)

Датасет 2 (SecLists, GitHub):
    SQL-payloads из SecLists/Fuzzing/SQLi
    Файл: seclists_sqli.txt

Использование:
    python src/download_data.py
"""

import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_seclists() -> None:
    """Скачивает SQL-payload файл из SecLists."""
    urls = [
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQLi/Generic-SQLi.txt",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQLi/quick-SQLi.txt",
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQL-Injection/Generic-SQLi.txt",
    ]
    out = DATA_RAW / "seclists_sqli.txt"
    if out.exists():
        print(f"[~] SecLists уже скачан: {out}")
        return

    print("[*] Скачиваю SecLists SQLi payloads...")
    text = None
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            text = r.text
            print(f"    URL: {url}")
            break
        except Exception:
            continue

    if text is None:
        print("[!] SecLists недоступен. Пропускаю — датасет соберётся без него.")
        return

    out.write_text(text, encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    print(f"[+] SecLists: {len(lines)} payloads → {out}")


def download_kaggle() -> None:
    """
    Скачивает датасет через Kaggle API.
    Требует: KAGGLE_USERNAME и KAGGLE_KEY в .env
    или файл ~/.kaggle/kaggle.json
    """
    out = DATA_RAW / "sqli_dataset.csv"
    if out.exists():
        print(f"[~] Kaggle датасет уже скачан: {out}")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[!] Установи kaggle: pip install kaggle")
        return

    print("[*] Скачиваю Kaggle датасет...")
    os.system(
        f'kaggle datasets download -d sajid576/sql-injection-dataset '
        f'--path "{DATA_RAW}" --unzip'
    )

    # Kaggle может скачать под другим именем — ищем CSV
    csvs = list(DATA_RAW.glob("*.csv"))
    if csvs and not out.exists():
        csvs[0].rename(out)
        print(f"[+] Переименован → {out}")

    if out.exists():
        import pandas as pd
        df = pd.read_csv(out, on_bad_lines="skip")
        print(f"[+] Kaggle: {len(df)} строк, колонки: {list(df.columns)}")
    else:
        print("[!] CSV не найден. Скачай вручную с kaggle.com и положи в data/raw/sqli_dataset.csv")
        print("    Ссылка: https://www.kaggle.com/datasets/sajid576/sql-injection-dataset")


def main() -> None:
    download_seclists()
    download_kaggle()
    print("\n[+] Готово. Теперь запусти: python src/dataset.py")


if __name__ == "__main__":
    main()
