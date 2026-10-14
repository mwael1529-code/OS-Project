"""Run post-training independent evaluation and parser integration smoke checks."""
from pathlib import Path
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from nlp_runtime import predict_intent, normalize_text
from voice_ubuntu_parser import parse_text, build_youtube_search_url


def main():
    base = Path(__file__).resolve().parent
    train = pd.read_csv(base / "training_examples.csv")
    heldout = pd.read_csv(base / "independent_eval_v6.csv")
    train_norm = {normalize_text(x) for x in train["text"].astype(str)}
    heldout["normalized"] = heldout["text"].map(normalize_text)
    overlap = heldout[heldout["normalized"].isin(train_norm)]
    if len(overlap):
        raise RuntimeError(
            "Independent evaluation contains exact normalized training overlap: "
            + repr(overlap["text"].tolist())
        )

    y_true, y_pred, rows = [], [], []
    for row in heldout.itertuples():
        prediction = predict_intent(row.text)
        y_true.append(row.intent)
        y_pred.append(prediction["intent"])
        rows.append({
            "text": row.text,
            "expected": row.intent,
            "actual": prediction["intent"],
            "score": prediction.get("score"),
            "passed": row.intent == prediction["intent"],
        })

    independent = {
        "n_examples": int(len(heldout)),
        "exact_normalized_training_overlap": int(len(overlap)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix_labels": sorted(set(y_true) | set(y_pred)),
        "rows": rows,
    }
    independent["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=independent["confusion_matrix_labels"]
    ).tolist()

    mock_apps = [{
        "display_name": "Firefox",
        "desktop_id": "firefox.desktop",
        "desktop_file": "/mock/firefox.desktop",
    }]
    parser_cases = [
        ("English YouTube search", "Search YouTube for Python tutorials", "SEARCH_YOUTUBE", "Python tutorials"),
        ("Arabic YouTube search", "ابحث على يوتيوب عن شرح التفاضل", "SEARCH_YOUTUBE", "شرح التفاضل"),
        ("Egyptian Arabic open YouTube", "ممكن تفتح يوتيوب", "OPEN_YOUTUBE", None),
        ("Local movie playback", "Play Interstellar", "PLAY_MEDIA", "Interstellar"),
        ("Arabic local song playback", "شغل أغنية تملي معاك", "PLAY_MEDIA", "تملي معاك"),
        ("Explicit YouTube playback search", "Play Interstellar on YouTube", "SEARCH_YOUTUBE", "Interstellar"),
        ("File target extraction", "Locate my project PDF", "SEARCH_FILE", "project PDF"),
        ("Folder target extraction", "Create a folder called Semester 4", "CREATE_FOLDER", "Semester 4"),
        ("Folder parent extraction", "Create a folder in Documents called Notes", "CREATE_FOLDER", "Notes"),
        ("Missing folder name clarification", "Create a new folder", "CREATE_FOLDER", None),
        ("Unsupported request", "Tell me a joke", "UNKNOWN", None),
        ("Compound command", "Open Firefox and search YouTube for rock and roll", "OPEN_APP", None),
    ]
    parser_rows = []
    for name, text, expected_intent, expected_entity in parser_cases:
        result = parse_text(text, applications=mock_apps)
        actions = result.get("actions", [])
        first = actions[0] if actions else {}
        actual_intents = [a.get("intent") for a in actions]
        passed = bool(actions) and first.get("intent") == expected_intent
        if expected_entity is not None:
            if expected_intent in {"SEARCH_YOUTUBE", "SEARCH_FILE"}:
                entity_value = first.get("entities", {}).get("query")
            elif expected_intent == "PLAY_MEDIA":
                entity_value = first.get("entities", {}).get("title")
            elif expected_intent == "CREATE_FOLDER":
                entity_value = first.get("entities", {}).get("folder_name")
            else:
                entity_value = None
            passed = passed and entity_value == expected_entity
        if name == "Compound command":
            passed = passed and actual_intents == ["OPEN_APP", "SEARCH_YOUTUBE"]
            if len(actions) > 1:
                passed = passed and actions[1].get("entities", {}).get("query") == "rock and roll"
        if name == "Missing folder name clarification":
            passed = passed and first.get("status") == "needs_clarification"
        parser_rows.append({
            "name": name, "input": text, "expected_intent": expected_intent,
            "actual_intents": actual_intents, "passed": bool(passed), "output": result,
        })

    # Verify round-trip encoding for multilingual/free-text YouTube topics.
    query_cases = [
        "rock and roll", "C++ pointers and references", "Linux; permissions & ownership",
        "شرح التفاضل والتكامل", "مقارنة بين SQL و Python", "café & crème brûlée",
    ]
    url_rows = []
    for query in query_cases:
        decoded = parse_qs(urlparse(build_youtube_search_url(query)).query).get("search_query", [])
        url_rows.append({"query": query, "round_trip": decoded == [query], "decoded": decoded})

    parser_passed = all(r["passed"] for r in parser_rows)
    url_passed = all(r["round_trip"] for r in url_rows)
    independent_passed = all(row["passed"] for row in rows)

    results = {
        "project": "Voice Ubuntu NLP",
        "version": "v6",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(train)),
        "training_intents": int(train["intent"].nunique()),
        "independent_eval": independent,
        "independent_eval_all_correct": independent_passed,
        "parser_smoke_tests": parser_rows,
        "parser_smoke_all_passed": parser_passed,
        "url_round_trip_tests": url_rows,
        "url_round_trip_all_passed": url_passed,
        "limitations": [
            "The independent evaluation contains 40 manually curated text commands, not independently collected microphone transcripts.",
            "A random row split may be optimistic because template-generated phrases can share structure.",
            "Firefox application data are mocked in parser smoke tests.",
            "No Ubuntu operation or shell command is executed by this evaluator.",
        ],
    }
    (base / "final_validation_v6.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "VOICE UBUNTU NLP v6 — FINAL VALIDATION REPORT",
        f"UTC timestamp: {results['timestamp_utc']}",
        f"Training rows: {len(train)}",
        f"Intent classes: {train['intent'].nunique()}",
        "",
        "INDEPENDENT EVALUATION",
        f"Rows: {len(heldout)}",
        f"Exact normalized overlap with training data: {len(overlap)}",
        f"Accuracy: {independent['accuracy']:.4f}",
        f"Macro F1: {independent['macro_f1']:.4f}",
        f"Weighted F1: {independent['weighted_f1']:.4f}",
        independent["classification_report"],
        "",
        "PARSER SMOKE TESTS",
    ]
    for row in parser_rows:
        report.append(
            f"{'PASS' if row['passed'] else 'FAIL'} | {row['name']} | "
            f"expected={row['expected_intent']} | actual={row['actual_intents']}"
        )
    report.extend(["", "YOUTUBE URL ROUND-TRIP TESTS"])
    for row in url_rows:
        report.append(f"{'PASS' if row['round_trip'] else 'FAIL'} | {row['query']}")
    report.extend([
        "",
        "LIMITATIONS",
        *results["limitations"],
        "",
        "The model returns structured intent data only. The security component must validate every action before the Ubuntu action component executes it.",
    ])
    (base / "FINAL_VALIDATION_REPORT_V6.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("Independent accuracy:", f"{independent['accuracy']:.1%}")
    print("Independent macro F1:", f"{independent['macro_f1']:.3f}")
    print("Exact training overlap:", len(overlap))
    print("Parser smoke tests:", sum(r["passed"] for r in parser_rows), "/", len(parser_rows))
    print("URL round trips:", sum(r["round_trip"] for r in url_rows), "/", len(url_rows))
    print("Saved:", base / "FINAL_VALIDATION_REPORT_V6.txt")
    print("Saved:", base / "final_validation_v6.json")

    if not (independent_passed and parser_passed and url_passed):
        raise SystemExit("One or more v6 evaluation checks failed. Review the report before packaging.")

if __name__ == "__main__":
    main()
