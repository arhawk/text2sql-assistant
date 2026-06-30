from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
PLACEHOLDER_RE = re.compile(r"<[A-Z_]+>")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_sql(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql.strip())
    sql = sql.replace(" ;", ";")
    return sql


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def sql_to_template(sql: str) -> str:
    template = re.sub(r"'[^']*'", "<STR>", sql)
    template = re.sub(r'"[^"]*"', "<STR>", template)
    template = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", template)
    return normalize_sql(template)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=json_default)


def save_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            elif is_dataclass(row):
                row = asdict(row)
            fh.write(json.dumps(row, ensure_ascii=False, default=json_default))
            fh.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("data")
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Unsupported JSON structure in {path}")


def text_from_example(example: dict[str, Any]) -> str:
    question = example.get("question") or example.get("text") or example.get("utterance")
    if not question:
        raise KeyError("Example is missing a question/text field")
    return str(question)


def sql_from_example(example: dict[str, Any]) -> str:
    sql = example.get("sql") or example.get("query")
    if sql is None:
        raise KeyError("Example is missing a sql/query field")
    return str(sql)


def split_name(example: dict[str, Any], default: str = "train") -> str:
    return str(example.get("split") or example.get("data") or default)


def iter_top_k_scored(items: Iterable[tuple[float, Any]], k: int) -> Iterator[Any]:
    ranked = sorted(items, key=lambda item: item[0], reverse=True)
    for _, item in ranked[:k]:
        yield item
