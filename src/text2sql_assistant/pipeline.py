from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .a4_models import GenerationModel, LLMEnsembleModel, TemplateClassificationModel
from .baseline_a2 import BaselineA2Model
from .common import ensure_dir, normalize_sql, save_json, save_jsonl, timestamp_slug
from .data import ensure_template_sql, split_examples
from .types import EvalSummary, Example, Prediction


@dataclass
class ModelBundle:
    baseline: BaselineA2Model
    classification: TemplateClassificationModel
    generation: GenerationModel
    llm: LLMEnsembleModel


def build_models(train_examples: list[Example]) -> ModelBundle:
    baseline = BaselineA2Model(train_examples)
    classification = TemplateClassificationModel(train_examples)
    generation = GenerationModel(train_examples)
    llm = LLMEnsembleModel(baseline, classification)
    return ModelBundle(
        baseline=baseline,
        classification=classification,
        generation=generation,
        llm=llm,
    )


def predict_one(question: str, model_name: str, models: ModelBundle) -> Prediction:
    registry = {
        "baseline": models.baseline,
        "classification": models.classification,
        "generation": models.generation,
        "llm": models.llm,
    }
    if model_name == "all":
        raise ValueError("predict_one does not support mode=all")
    if model_name not in registry:
        raise KeyError(f"Unknown model: {model_name}")
    return registry[model_name].predict(question)


def predict_all(question: str, models: ModelBundle) -> list[Prediction]:
    return [
        models.baseline.predict(question),
        models.classification.predict(question),
        models.generation.predict(question),
        models.llm.predict(question),
    ]


def evaluate(models: ModelBundle, test_examples: list[Example], model_name: str) -> tuple[EvalSummary, list[Prediction]]:
    pred_rows: list[Prediction] = []
    exact = 0
    total = len(test_examples)

    for example in test_examples:
        pred = predict_one(example.question, model_name, models)
        pred_rows.append(pred)
        if normalize_sql(pred.sql) == normalize_sql(example.sql):
            exact += 1

    run_dir = ensure_dir(Path("artifacts") / "runs" / f"{timestamp_slug()}-{model_name}")
    save_jsonl(run_dir / "predictions.jsonl", [row.to_dict() for row in pred_rows])
    errors = [
        {
            "question": example.question,
            "gold_sql": normalize_sql(example.sql),
            "pred_sql": normalize_sql(pred.sql),
            "mode": pred.mode,
            "notes": pred.notes,
        }
        for example, pred in zip(test_examples, pred_rows)
        if normalize_sql(example.sql) != normalize_sql(pred.sql)
    ]
    save_jsonl(run_dir / "errors.jsonl", errors)

    summary = EvalSummary(
        mode=model_name,
        total=total,
        exact_match=exact,
        accuracy=(exact / total) if total else 0.0,
        artifact_dir=str(run_dir),
    )
    save_json(run_dir / "summary.json", summary.to_dict())
    return summary, pred_rows


def prepare_split_dataset(examples: list[Example]) -> tuple[list[Example], list[Example], list[Example]]:
    examples = ensure_template_sql(examples)
    train = split_examples(examples, "train")
    dev = split_examples(examples, "dev")
    test = split_examples(examples, "test")
    return train, dev, test


def export_dataset_split(examples: list[Example], output_root: str | Path) -> Path:
    output_root = ensure_dir(output_root)
    train, dev, test = prepare_split_dataset(examples)
    save_jsonl(output_root / "train.jsonl", [ex.to_dict() for ex in train])
    save_jsonl(output_root / "dev.jsonl", [ex.to_dict() for ex in dev])
    save_jsonl(output_root / "test.jsonl", [ex.to_dict() for ex in test])
    return output_root
