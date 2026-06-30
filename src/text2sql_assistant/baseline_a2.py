from __future__ import annotations

from dataclasses import dataclass

from .common import jaccard, normalize_sql, tokenize
from .types import Example, Prediction


@dataclass
class BaselineA2Model:
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
                mode="baseline_a2",
                question=question,
                sql="SELECT 1;",
                score=0.0,
                matched_question="",
                matched_sql="SELECT 1;",
                notes=["fallback"],
            )

        return Prediction(
            mode="baseline_a2",
            question=question,
            sql=normalize_sql(best_example.sql),
            score=best_score,
            matched_question=best_example.question,
            matched_sql=normalize_sql(best_example.sql),
            notes=["nearest-neighbor retrieval"],
        )
