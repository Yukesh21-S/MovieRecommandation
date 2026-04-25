"""Run end-to-end evaluation for MovieRecommandation chatbot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movie_chatbot.agent.nodes import (
    chat,
    classify,
    fetch_discover,
    followup,
    respond_discover,
    respond_search,
    retrieve,
    search_movie,
)
from movie_chatbot.agent.state import State
from movie_chatbot.services import data_fetcher as tmdb
from movie_chatbot.services import vector_engine as vdb


DATASET_PATH = ROOT / "evaluation" / "qa_dataset.json"
OUTPUT_DIR = ROOT / "evaluation" / "outputs"


def _title_tokens(text: str) -> list[str]:
    stop = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or"}
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok not in stop]


def _near_exact_title_match(actual: str, expected: str) -> bool:
    actual_tokens = _title_tokens(actual)
    expected_tokens = _title_tokens(expected)
    if not actual_tokens or not expected_tokens:
        return False
    if actual_tokens == expected_tokens:
        return True
    if expected_tokens == actual_tokens[: len(expected_tokens)]:
        return True
    token_ratio = len(actual_tokens) / max(len(expected_tokens), 1)
    return set(expected_tokens).issubset(set(actual_tokens)) and token_ratio <= 1.25


def ensure_db_ready() -> None:
    collection = vdb.get_collection()
    if collection and collection.count() > 0:
        print(f"[Eval Setup] Vector DB ready ({collection.count()} movies).")
        return
    print("[Eval Setup] Building vector DB for evaluation...")
    data_path = ROOT / "movies_data.json"
    if not data_path.exists():
        tmdb.fetch_and_save_diverse_movies(str(data_path))
    vdb.setup_vector_db(str(data_path))


def init_state(query: str, history: list[dict] | None = None) -> State:
    return {
        "query": query,
        "history": history or [],
        "intent": "discover",
        "search_title": None,
        "language_code": None,
        "genre_id": None,
        "person": None,
        "is_latest": False,
        "is_lite_mood": False,
        "movies": [],
        "disambiguation_options": None,
        "response": "",
    }


def run_one(query: str, history: list[dict] | None = None) -> dict[str, Any]:
    state = classify(init_state(query, history))
    intent = state["intent"]

    if intent == "chat":
        state = chat(state)
    elif intent == "followup":
        state = followup(state)
    elif intent == "search":
        state = search_movie(state)
        state = respond_search(state)
    else:
        state = retrieve(state)
        if len(state["movies"]) < 3:
            state = fetch_discover(state)
        state = respond_discover(state)

    retrieved_docs = [
        {
            "title": movie.get("title"),
            "genres": movie.get("genres"),
            "release_date": movie.get("release_date"),
            "vote_average": movie.get("vote_average"),
        }
        for movie in state.get("movies", [])
    ]
    return {
        "intent": intent,
        "retrieved_docs": retrieved_docs,
        "final_answer": state.get("response", ""),
    }


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = case.get("checks", {})
    answer_lower = result["final_answer"].lower()
    retrieved_titles = [str(doc.get("title", "")).lower() for doc in result["retrieved_docs"]]
    reasons: list[str] = []
    passed = True

    min_retrieved = checks.get("min_retrieved")
    if isinstance(min_retrieved, int):
        ok = len(result["retrieved_docs"]) >= min_retrieved
        passed = passed and ok
        reasons.append(f"min_retrieved>={min_retrieved}: {'PASS' if ok else 'FAIL'}")

    answer_keywords = checks.get("answer_keywords_any")
    if isinstance(answer_keywords, list) and answer_keywords:
        ok = any(keyword.lower() in answer_lower for keyword in answer_keywords)
        passed = passed and ok
        reasons.append(f"answer_keywords_any: {'PASS' if ok else 'FAIL'}")

    title_keywords = checks.get("title_keywords_any")
    if isinstance(title_keywords, list) and title_keywords:
        top_title = str(result["retrieved_docs"][0].get("title", "")).lower() if result["retrieved_docs"] else ""
        ok = any(
            _near_exact_title_match(top_title, keyword.lower())
            for keyword in title_keywords
        )
        passed = passed and ok
        reasons.append(f"title_keywords_any: {'PASS' if ok else 'FAIL'}")

    return passed, reasons


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of Q&A cases.")
    return data


def save_reports(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"eval_log_{stamp}.jsonl"
    summary_path = OUTPUT_DIR / f"eval_summary_{stamp}.json"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"[Eval] Detailed log: {jsonl_path}")
    print(f"[Eval] Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model evaluation pipeline.")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Path to Q&A dataset JSON.")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    ensure_db_ready()

    records: list[dict[str, Any]] = []
    passed_count = 0
    for case in dataset:
        try:
            result = run_one(case["query"], history=case.get("history"))
            passed, reasons = evaluate_case(case, result)
            if passed:
                passed_count += 1
        except Exception as exc:  # noqa: BLE001
            result = {"intent": "error", "retrieved_docs": [], "final_answer": ""}
            passed = False
            reasons = [f"runtime_error: {type(exc).__name__}: {exc}"]

        record = {
            "id": case["id"],
            "query": case["query"],
            "expected": case.get("checks", {}),
            "predicted_intent": result["intent"],
            "retrieved_docs": result["retrieved_docs"],
            "final_answer": result["final_answer"],
            "passed": passed,
            "reasons": reasons,
        }
        records.append(record)
        print(f"[{case['id']:02d}] {'PASS' if passed else 'FAIL'} - {case['query']}")

    total = len(dataset)
    accuracy = (passed_count / total * 100.0) if total else 0.0
    summary = {
        "total_cases": total,
        "passed_cases": passed_count,
        "failed_cases": total - passed_count,
        "accuracy_percent": round(accuracy, 2),
    }
    print(
        f"\n[Eval Result] Accuracy: {summary['accuracy_percent']}% "
        f"({summary['passed_cases']}/{summary['total_cases']})"
    )
    save_reports(records, summary)


if __name__ == "__main__":
    main()

