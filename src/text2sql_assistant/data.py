from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .common import load_json_or_jsonl, sql_from_example, split_name, text_from_example
from .types import Example


def load_examples(path: str | Path) -> list[Example]:
    rows = load_json_or_jsonl(path)
    examples: list[Example] = []
    for row in rows:
        examples.append(
            Example(
                question=text_from_example(row),
                sql=sql_from_example(row),
                split=split_name(row),
                source=str(row.get("source", "loaded")),
                template_sql=row.get("template_sql"),
                extra={k: v for k, v in row.items() if k not in {"question", "text", "utterance", "sql", "query", "split", "data", "source", "template_sql"}},
            )
        )
    return examples


def save_examples(path: str | Path, examples: Iterable[Example]) -> None:
    from .common import save_jsonl

    save_jsonl(path, [ex.to_dict() for ex in examples])


def split_examples(examples: list[Example], split: str) -> list[Example]:
    return [example for example in examples if example.split == split]


def ensure_template_sql(examples: list[Example]) -> list[Example]:
    from .common import sql_to_template

    normalized: list[Example] = []
    for example in examples:
        template_sql = example.template_sql or sql_to_template(example.sql)
        normalized.append(
            Example(
                question=example.question,
                sql=example.sql,
                split=example.split,
                source=example.source,
                template_sql=template_sql,
                extra=example.extra,
            )
        )
    return normalized
