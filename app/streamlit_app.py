from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql_assistant.common import normalize_sql
from text2sql_assistant.data import load_examples
from text2sql_assistant.pipeline import build_models, evaluate, predict_all, predict_one


st.set_page_config(
    page_title="Text2SQL Assistant",
    page_icon="S",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str):
    examples = load_examples(REPO_ROOT / dataset_path)
    train_examples = [ex for ex in examples if ex.split == "train"]
    dev_examples = [ex for ex in examples if ex.split == "dev"]
    test_examples = [ex for ex in examples if ex.split == "test"]
    return examples, train_examples, dev_examples, test_examples


@st.cache_resource(show_spinner=False)
def build_model_bundle(train_examples):
    return build_models(train_examples)


def render_prediction_card(title: str, pred, gold_sql: str | None = None):
    with st.container(border=True):
        st.subheader(title)
        left, right = st.columns([2, 1])
        with left:
            st.code(normalize_sql(pred.sql), language="sql")
        with right:
            st.metric("score", f"{pred.score:.4f}")
            if gold_sql is not None:
                match = normalize_sql(pred.sql) == normalize_sql(gold_sql)
                st.metric("exact match", "yes" if match else "no")
        st.caption(f"matched question: {pred.matched_question or '-'}")
        if pred.notes:
            st.caption("notes: " + ", ".join(pred.notes))


def run_single_question(question: str, mode: str, models):
    if mode == "all":
        return predict_all(question, models)
    return [predict_one(question, mode, models)]


def main():
    st.title("Text2SQL Assistant")
    st.write("A demo that combines an A2-style baseline with A4-style adapters.")

    with st.sidebar:
        st.header("Settings")
        dataset_path = st.text_input("Dataset", value="data/sample_text2sql.jsonl")
        mode = st.selectbox(
            "Model",
            ["all", "baseline", "classification", "generation", "llm"],
            index=0,
        )
        st.caption("Use `all` to compare every model on one query.")

    try:
        examples, train_examples, dev_examples, test_examples = load_dataset(dataset_path)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    models = build_model_bundle(train_examples)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("train", len(train_examples))
    col2.metric("dev", len(dev_examples))
    col3.metric("test", len(test_examples))
    col4.metric("total", len(examples))

    tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Evaluation", "Dataset Preview"])

    with tab1:
        question = st.text_area(
            "Enter a natural language question",
            value="show flights from boston to denver",
            height=100,
        )
        run_btn = st.button("Run prediction", type="primary")

        if run_btn and question.strip():
            predictions = run_single_question(question.strip(), mode, models)
            if len(predictions) == 1:
                render_prediction_card(predictions[0].mode, predictions[0])
            else:
                for pred in predictions:
                    render_prediction_card(pred.mode, pred)

    with tab2:
        if not test_examples:
            st.info("No test split found in the loaded dataset.")
        else:
            eval_mode = st.selectbox(
                "Evaluation mode",
                ["baseline", "classification", "generation", "llm", "all"],
                index=4,
                key="eval_mode",
            )
            eval_btn = st.button("Run evaluation")

            if eval_btn:
                modes = ["baseline", "classification", "generation", "llm"] if eval_mode == "all" else [eval_mode]
                summary_rows = []
                for current_mode in modes:
                    summary, _ = evaluate(models, test_examples, current_mode)
                    summary_rows.append(summary.to_dict())
                    st.success(
                        f"{current_mode}: accuracy={summary.accuracy:.4f}, exact={summary.exact_match}/{summary.total}"
                    )

                st.dataframe(summary_rows, use_container_width=True, hide_index=True)
                st.caption("Each mode also writes its own artifact folder under `artifacts/runs/`.")

    with tab3:
        preview_rows = []
        for example in examples[:20]:
            preview_rows.append(
                {
                    "split": example.split,
                    "question": example.question,
                    "sql": normalize_sql(example.sql),
                    "template_sql": example.template_sql or "",
                }
            )
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
