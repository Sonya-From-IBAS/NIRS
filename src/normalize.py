"""
Нормализация SQL-запросов перед классификацией.
Декодирует hex, CHAR(), убирает лишние комментарии.
Используется как sklearn-трансформер в пайплайне.
"""

import re
from sklearn.base import BaseEstimator, TransformerMixin


def _decode_hex(match: re.Match) -> str:
    try:
        return bytes.fromhex(match.group(1)).decode("utf-8", errors="replace")
    except Exception:
        return match.group(0)


def _decode_char(match: re.Match) -> str:
    try:
        nums = re.findall(r"\d+", match.group(0))
        return "".join(chr(int(n)) for n in nums if 32 <= int(n) <= 126)
    except Exception:
        return match.group(0)


def normalize_sql(query: str) -> str:
    q = str(query)
    # 0x... → ASCII
    q = re.sub(r"0x([0-9a-fA-F]+)", _decode_hex, q, flags=re.IGNORECASE)
    # CHAR(65,66,67) → ABC
    q = re.sub(r"CHAR\s*\([\d,\s]+\)", _decode_char, q, flags=re.IGNORECASE)
    # убираем inline-комментарии /* ... */
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.DOTALL)
    # нормализуем URL-кодирование
    q = re.sub(r"%20", " ", q)
    q = re.sub(r"%27", "'", q)
    q = re.sub(r"%3D", "=", q)
    # несколько пробелов → один
    q = re.sub(r"\s+", " ", q).strip()
    return q


class SQLNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn-совместимый трансформер для нормализации SQL."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_sql(q) for q in X]
