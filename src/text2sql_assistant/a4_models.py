from __future__ import annotations

from dataclasses import dataclass
import re

from .common import jaccard, normalize_sql, sql_to_template, tokenize
from .types import Example, Prediction


def _extract_slots(question: str) -> dict[str, str]:
    q = question.strip().lower()
    slots: dict[str, str] = {}

    from_to = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\s+on\s+|\s+at\s+|\?|$)", q)
    if from_to:
        slots["<ORIGIN>"] = from_to.group(1).strip()
        slots["<DEST>"] = from_to.group(2).strip()

    between = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s+on\s+|\s+at\s+|\?|$)", q)
    if between:
        slots["<LEFT>"] = between.group(1).strip()
        slots["<RIGHT>"] = between.group(2).strip()

    day = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q)
    if day:
        slots["<DAY>"] = day.group(1)

    time = re.search(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", q)
    if time:
        raw = time.group(0).replace(" ", "")
        slots["<TIME>"] = f"'{raw}'" if ":" in raw or raw.endswith(("am", "pm")) else raw

    number = re.search(r"\b\d+\b", q)
    if number:
        slots["<NUM>"] = number.group(0)

    destination = re.search(r"\bto\s+(.+?)(?:\s+on\s+|\s+at\s+|\?|$)", q)
    if destination and "<DEST>" not in slots:
        slots["<DEST>"] = destination.group(1).strip()

    origin = re.search(r"\bfrom\s+(.+?)(?:\s+to\s+|\s+on\s+|\s+at\s+|\?|$)", q)
    if origin and "<ORIGIN>" not in slots:
        slots["<ORIGIN>"] = origin.group(1).strip()

    return slots


def _fill_template(template_sql: str, slots: dict[str, str]) -> tuple[str, list[str]]:
    filled = normalize_sql(template_sql)
    notes: list[str] = []
    for placeholder, value in slots.items():
        if placeholder in filled:
            if not re.fullmatch(r"\d+(?:\.\d+)?", value) and not value.startswith(("'", '"')):
                value = f"'{value}'"
            filled = filled.replace(placeholder, value)
            notes.append(f"filled {placeholder}")
    remaining = sorted(set(re.findall(r"<[A-Z_]+>", filled)))
    if remaining:
        notes.append(f"unfilled placeholders: {', '.join(remaining)}")
    return filled, notes


@dataclass
class TemplateClassificationModel:
    training_examples: list[Example]

    def predict(self, question: str) -> Prediction:
        q_tokens = tokenize(question)
        best_example = None
        best_score = -1.0

        for example in self.training_examples:
            score = jaccard(q_tokens, tokenize(example.question))
            if score > best_score:
                best_score = score
                best_example = example

        if best_example is None:
            return Prediction(
                mode="a4_classification",
                question=question,
                sql="SELECT 1;",
                score=0.0,
                matched_question="",
                matched_sql="SELECT 1;",
                notes=["fallback"],
            )

        template_sql = best_example.template_sql or sql_to_template(best_example.sql)
        slots = _extract_slots(question)
        filled_sql, notes = _fill_template(template_sql, slots)
        if filled_sql == template_sql:
            notes.append("template used without slot fill")

        return Prediction(
            mode="a4_classification",
            question=question,
            sql=filled_sql,
            score=best_score,
            matched_question=best_example.question,
            matched_sql=normalize_sql(best_example.sql),
            notes=notes + ["template classification"],
        )


@dataclass
class GenerationModel:
    training_examples: list[Example]

    def predict(self, question: str) -> Prediction:
        q_tokens = tokenize(question)
        scored = []
        for example in self.training_examples:
            base_score = jaccard(q_tokens, tokenize(example.question))
            template_bonus = 0.0
            if example.template_sql:
                template_bonus = 0.05 * len(set(re.findall(r"<[A-Z_]+>", example.template_sql)))
            scored.append((base_score + template_bonus, example))

        best_example = max(scored, key=lambda item: item[0], default=None)
        if best_example is None:
            return Prediction(
                mode="a4_generation",
                question=question,
                sql="SELECT 1;",
                score=0.0,
                matched_question="",
                matched_sql="SELECT 1;",
                notes=["fallback"],
            )

        score, example = best_example
        candidate_sql = normalize_sql(example.sql)
        if example.template_sql:
            filled_sql, notes = _fill_template(example.template_sql, _extract_slots(question))
            if filled_sql.count("<") == 0:
                candidate_sql = filled_sql
                notes.append("template generation")
            else:
                notes.append("partial template generation")
        else:
            notes = ["retrieved exact sql"]

        return Prediction(
            mode="a4_generation",
            question=question,
            sql=candidate_sql,
            score=score,
            matched_question=example.question,
            matched_sql=normalize_sql(example.sql),
            notes=notes,
        )


@dataclass
class LLMEnsembleModel:
    baseline_model: object
    classification_model: object

    def predict(self, question: str) -> Prediction:
        baseline_pred = self.baseline_model.predict(question)
        classification_pred = self.classification_model.predict(question)

        if normalize_sql(baseline_pred.sql) == normalize_sql(classification_pred.sql):
            return Prediction(
                mode="a4_llm",
                question=question,
                sql=normalize_sql(classification_pred.sql),
                score=max(baseline_pred.score, classification_pred.score),
                matched_question=classification_pred.matched_question,
                matched_sql=classification_pred.matched_sql,
                notes=["ensemble agreement"],
            )

        preferred = classification_pred if classification_pred.score >= baseline_pred.score else baseline_pred
        return Prediction(
            mode="a4_llm",
            question=question,
            sql=normalize_sql(preferred.sql),
            score=preferred.score,
            matched_question=preferred.matched_question,
            matched_sql=preferred.matched_sql,
            notes=[
                "ensemble fallback",
                f"baseline={normalize_sql(baseline_pred.sql)}",
                f"classification={normalize_sql(classification_pred.sql)}",
            ],
        )
