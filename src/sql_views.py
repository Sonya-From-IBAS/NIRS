"""Lossless SQL lexical views, identifier masking and pre-transformation signals.

This is a tolerant lexer for research features, not a dialect SQL parser.
Quoted text, comments and procedural dollar bodies remain separate tokens.
"""

from collections import Counter
from dataclasses import dataclass
import math
import re

VERSION = "sql-views-v1"
KEYWORDS = set("""SELECT FROM WHERE AND OR NOT NULL IS IN EXISTS BETWEEN LIKE ILIKE ESCAPE
UNION ALL DISTINCT JOIN INNER OUTER LEFT RIGHT FULL CROSS ON USING AS WITH RECURSIVE
GROUP BY HAVING ORDER ASC DESC LIMIT OFFSET FETCH FIRST NEXT ROW ROWS ONLY TOP
INSERT INTO VALUES UPDATE SET DELETE MERGE WHEN THEN ELSE END CASE RETURNING OUTPUT
CREATE ALTER DROP TRUNCATE TABLE VIEW INDEX DATABASE SCHEMA USER ROLE PROCEDURE FUNCTION
TRIGGER GRANT REVOKE TO PUBLIC PRIVILEGES OPTION ADMIN MEMBER ADD EXEC EXECUTE CALL DO
BEGIN DECLARE RETURN LANGUAGE SECURITY DEFINER INVOKER OWNER AUTHID SQL CURRENT_USER
SESSION_USER CURRENT_ROLE IDENTIFIED PASSWORD AUTHORIZATION CASCADE RESTRICT DEFAULT
COPY OUTFILE DUMPFILE PROGRAM CSV HEADER FORMAT DELIMITER COMMIT ROLLBACK SAVEPOINT
INT INTEGER BIGINT SMALLINT DECIMAL NUMERIC FLOAT DOUBLE PRECISION REAL TEXT VARCHAR
CHARACTER BOOLEAN TRUE FALSE DATE TIMESTAMP INTERVAL CAST CONVERT UNSIGNED COLLATE
OVER PARTITION ROW_NUMBER RANK DENSE_RANK FILTER WINDOW EXCEPT INTERSECT FOR OF
IF LOOP WHILE TRY CATCH TRANSACTION TEMP TEMPORARY MATERIALIZED EXPLAIN ANALYZE
SHOW DESCRIBE DESC USE NOLOCK LOCK NOWAIT SKIP LOCKED OBJECT GRANTED INCLUDE
CHECK PRIMARY KEY FOREIGN REFERENCES UNIQUE CONSTRAINT COUNT SUM AVG MIN MAX
COALESCE NULLIF NVL DECODE SUBSTRING SUBSTR TRIM LENGTH OCTET_LENGTH LOWER UPPER
CONCAT CHAR CHR ASCII UNICODE UNHEX HEX ENCODE DECODE CONVERT_TO TO_NUMBER TO_CHAR
SLEEP BENCHMARK WAITFOR DELAY PG_SLEEP LOAD_FILE PG_READ_FILE PG_READ_BINARY_FILE
PG_LS_DIR LO_EXPORT DBLINK OPENROWSET OPENQUERY XP_CMDSHELL XP_DIRTREE
UTL_HTTP UTL_FILE UTL_INADDR DBMS_SCHEDULER SYS_CONTEXT SYSTEM_USER SP_EXECUTESQL
SYSUTCDATETIME GETDATE CURRENT_TIMESTAMP CURRENT_DATE SYSDATE DUAL""".split())

TOKEN = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<exec_comment>/\*[!+][\s\S]*?(?:\*/|$))"
    r"|(?P<block_comment>/\*[\s\S]*?(?:\*/|$))"
    r"|(?P<line_comment>--[^\r\n]*|\#[^\r\n]*)"
    r"|(?P<dollar>\$(?P<tag>[A-Za-z_][A-Za-z_0-9]*|)\$[\s\S]*?\$(?P=tag)\$)"
    r"|(?P<string>(?:[nNeE])?'(?:''|\\.|[^'\\])*')"
    r'|(?P<quoted_identifier>"(?:""|[^"])*"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\])'
    r"|(?P<hex>0[xX][0-9A-Fa-f]+|[xX]'[0-9A-Fa-f]*')"
    r"|(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<word>(?:[^\W\d]|_)[\w$]*)"
    r"|(?P<operator><=|>=|<>|!=|\|\||::|:=|.)",
    re.DOTALL,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    start: int
    end: int


def lex(sql):
    return [Token(match.lastgroup, match.group(), match.start(), match.end()) for match in TOKEN.finditer(sql)]


class Symbols:
    """Shared SQL/context symbols preserve equality without retaining names."""
    def __init__(self):
        self.names, self.values = {}, {}

    def identifier(self, value, quoted=False):
        key = ("quoted" if quoted else "plain", value if quoted else value.lower())
        if key not in self.names:
            self.names[key] = f"ID{len(self.names)}"
        return self.names[key]

    def literal(self, value, kind="STR"):
        key = (kind, str(value))
        if key not in self.values:
            self.values[key] = f"{kind}{len(self.values)}"
        return self.values[key]

    def resource(self, value):
        text = str(value)
        if "@" in text and "'" not in text:
            return "@".join(self.reference(part) for part in text.split("@", 1))
        if "/" in text or "\\" in text or "://" in text:
            return self.literal(text)
        pieces = [token for token in lex(text) if token.kind != "space"]
        if pieces and all(token.kind in ("word", "quoted_identifier") or token.text == "." for token in pieces):
            return "".join(("Q_" + self.identifier(token.text[1:-1], quoted=True) if token.kind == "quoted_identifier" else
                            self.reference(token.text) if token.kind == "word" else token.text) for token in pieces)
        return ".".join(self.reference(part) for part in text.split("."))

    def reference(self, value):
        if ("STR", value) in self.values:
            return self.values[("STR", value)]
        return self.identifier(value)


def string_value(text):
    if text[:1].upper() in ("N", "E"):
        text = text[1:]
    return text[1:-1].replace("''", "'")


def masked_tokens(tokens, symbols):
    output = []
    for token in tokens:
        kind, value = token.kind, token.text
        if kind == "space":
            continue
        if kind in ("block_comment", "line_comment"):
            output.append("COMMENT")
        elif kind == "exec_comment":
            output.extend(["EXEC_COMMENT", *masked_tokens(lex(value[3:-2] if value.endswith("*/") else value[3:]), symbols), "END_EXEC_COMMENT"])
        elif kind == "dollar":
            delimiter = re.match(r"\$[A-Za-z_0-9]*\$", value).group()
            # Preserve operation structure inside procedural bodies; retain the
            # delimiter category without learning its arbitrary user-chosen tag.
            output.extend(["DOLLAR_BODY", *masked_tokens(lex(value[len(delimiter):-len(delimiter)]), symbols), "END_DOLLAR_BODY"])
        elif kind == "word":
            output.append(value.upper() if value.upper() in KEYWORDS else symbols.identifier(value))
        elif kind == "quoted_identifier":
            delimiter = value[0]
            closing = "]" if delimiter == "[" else delimiter
            decoded = value[1:-1].replace(closing * 2, closing)
            output.append("Q_" + symbols.identifier(decoded, quoted=True))
        elif kind == "string":
            output.append(symbols.literal(string_value(value)))
        elif kind in ("number", "hex"):
            output.append(symbols.literal(value.lower(), "NUM" if kind == "number" else "HEX"))
        else:
            output.append(value)
    return output


def clause_parts(tokens, rendered):
    # A top-level WHERE partitions the original token stream. Nested WHEREs
    # remain in their enclosing segment; operators outside WHERE are retained.
    depth, where, suffix = 0, None, None
    for index, token in enumerate(tokens):
        if token.kind == "operator":
            depth += token.text == "("
            depth -= token.text == ")"
        if token.kind == "word" and depth == 0:
            word = token.text.upper()
            if word == "WHERE" and where is None:
                where = index
            elif where is not None and suffix is None and word in {"GROUP", "HAVING", "ORDER", "LIMIT", "UNION", "RETURNING"}:
                suffix = index
    if where is None:
        return "SEG_BODY " + " ".join(rendered) + " SEG_NO_OUTER_WHERE"
    # Render with the same symbol mapping/order as the whole text.
    segments = []
    for name, start, stop in (("SEG_BODY", 0, where), ("SEG_WHERE", where, suffix or len(tokens)),
                              ("SEG_TAIL", suffix or len(tokens), len(tokens))):
        segments.extend([name, *rendered[start:stop]])
    return " ".join(segments)


def structural_signals(sql, tokens):
    result = Counter()
    result["characters"] = len(sql)
    result["single_quotes_raw"] = sql.count("'")
    result["double_quotes_raw"] = sql.count('"')
    result["url_escape_markers_raw"] = len(re.findall(r"%[0-9a-fA-F]{2}", sql))
    significant = [token for token in tokens if token.kind not in ("space", "block_comment", "line_comment")]
    depth, maximum = 0, 0
    for index, token in enumerate(significant):
        result["token_" + token.kind] += 1
        if token.kind == "word" and token.text.upper() in KEYWORDS:
            result["keyword_" + token.text.upper()] += 1
        elif token.kind == "operator":
            if token.text in {"=", "<", ">", "<=", ">=", "!=", "<>", ";", "||", "+", "*"}:
                result["operator_" + token.text] += 1
            depth += token.text == "("
            depth -= token.text == ")"
            maximum = max(maximum, depth)
            if token.text in ("=", "!=", "<>") and index and index + 1 < len(significant):
                left, right = significant[index - 1], significant[index + 1]
                if left.kind in ("number", "string", "hex") and right.kind == left.kind:
                    result["comparison_same_literal" if left.text == right.text else "comparison_different_literal"] += 1
    result["max_parenthesis_depth"] = maximum
    result["parenthesis_imbalance"] = abs(depth)
    result["unclosed_single_quote_hint"] = int(sql.count("'") % 2 != 0)
    result["block_comment_count"] = sum(token.kind == "block_comment" for token in tokens)
    result["line_comment_count"] = sum(token.kind == "line_comment" for token in tokens)
    result["executable_comment_count"] = sum(token.kind == "exec_comment" for token in tokens)
    result["comment_characters"] = sum(len(token.text) for token in tokens if "comment" in token.kind)
    result["char_function_markers_raw"] = len(re.findall(r"\b(?:CHAR|CHR)\s*\(", sql, re.I))
    result["concat_markers_raw"] = len(re.findall(r"\|\||\bCONCAT\s*\(", sql, re.I))
    result["hex_markers_raw"] = len(re.findall(r"\b0x[0-9a-f]+|\bX'[0-9a-f]+'", sql, re.I))
    result["catalog_markers_raw"] = len(re.findall(r"information_schema|mysql\.user|pg_shadow|sys\.all_", sql, re.I))
    return {key: math.log1p(value) for key, value in result.items()}


def fixed_string_hints(tokens, symbols, depth=0):
    """Describe fixed SQL-looking strings; never evaluate or replace source SQL."""
    if depth > 2:
        return []
    significant = [token for token in tokens if token.kind not in ("space", "block_comment", "line_comment")]
    result, index = [], 0
    while index < len(significant):
        token = significant[index]
        if token.kind == "dollar":
            delimiter = re.match(r"\$[A-Za-z_0-9]*\$", token.text).group()
            result.extend(fixed_string_hints(lex(token.text[len(delimiter):-len(delimiter)]), symbols, depth + 1))
        if token.kind != "string":
            index += 1
            continue
        values, stop = [string_value(token.text)], index + 1
        while stop + 1 < len(significant) and significant[stop].text in ("||", "+") and significant[stop + 1].kind == "string":
            values.append(string_value(significant[stop + 1].text))
            stop += 2
        prefix = {item.text.upper() for item in significant[max(0, index - 3):index] if item.kind == "word"}
        text = "".join(values)
        if (len(values) > 1 or prefix & {"EXEC", "EXECUTE", "IMMEDIATE", "PREPARE"}) and len(text) <= 4096 and re.match(r"\s*(?:GRANT|REVOKE|SELECT|CALL|EXEC|ALTER|COPY|CREATE)\b", text, re.I):
            result.extend(["FIXED_SQL_TEXT_HINT", *masked_tokens(lex(text), symbols), "END_FIXED_HINT"])
        index = stop
    return result


def scalar_text_hints(tokens, symbols, depth=0):
    if depth > 2:
        return []
    items = [token for token in tokens if token.kind not in ("space", "block_comment", "line_comment")]
    def character(start):
        if start + 3 >= len(items) or items[start].text.upper() not in {"CHAR", "CHR"} or items[start + 1].text != "(":
            return None
        position, codes = start + 2, []
        while position < len(items) and items[position].kind == "number" and items[position].text.isdigit():
            number = int(items[position].text)
            if number < 32 or number > 126:
                return None
            codes.append(number); position += 1
            if position < len(items) and items[position].text == ",":
                position += 1
                continue
            break
        if not codes or len(codes) > 128 or position >= len(items) or items[position].text != ")":
            return None
        return "".join(chr(number) for number in codes), position + 1
    result, index = [], 0
    while index < len(items):
        token = items[index]
        if token.kind == "dollar":
            delimiter = re.match(r"\$[A-Za-z_0-9]*\$", token.text).group()
            result.extend(scalar_text_hints(lex(token.text[len(delimiter):-len(delimiter)]), symbols, depth + 1))
        is_concat = token.text.upper() == "CONCAT" and index + 1 < len(items) and items[index + 1].text == "("
        found = character(index + 2 if is_concat else index)
        if not found:
            index += 1
            continue
        text, stop = found
        while stop + 1 < len(items) and items[stop].text in ({","} if is_concat else {"||", "+"}):
            next_value = character(stop + 1)
            if not next_value:
                break
            fragment, stop = next_value
            text += fragment
        if len(text) <= 128:
            if text.isdigit():
                result.extend(["SCALAR_DIGITS_HINT", symbols.literal(str(int(text)), "NUM")])
            else:
                result.extend(["SCALAR_TEXT_HINT", symbols.literal(text)])
        index = stop
    return result


def views(sql, symbols=None):
    if not isinstance(sql, str):
        raise TypeError("SQL input must be a string")
    symbols = symbols or Symbols()
    tokens = lex(sql)
    visible = [token for token in tokens if token.kind != "space"]
    # One rendered token per source token for deterministic clause boundaries.
    rendered = [" ".join(masked_tokens([token], symbols)) for token in visible]
    hints = " ".join(fixed_string_hints(tokens, symbols) + scalar_text_hints(tokens, symbols))
    return {"raw": sql, "masked": (" ".join(rendered) + " " + hints).strip() or "EMPTY_SQL",
            "parts": (clause_parts(visible, rendered) + " " + hints).strip(),
            "structure": structural_signals(sql, tokens), "symbols": symbols}
