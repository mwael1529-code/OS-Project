"""Train Voice Ubuntu NLP v6 from training_examples.csv."""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from nlp_runtime import normalize_text


def make_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                     sublinear_tf=True, max_features=20000)),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                          sublinear_tf=True, max_features=35000)),
        ])),
        ("classifier", LogisticRegression(max_iter=4000, class_weight="balanced",
                                           C=2.5, random_state=42)),
    ])


def load_dataset(base: Path):
    data = pd.read_csv(base / "training_examples.csv")
    required = {"text", "intent"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}")
    data = data[["text", "intent"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data[data["text"].ne("") & data["intent"].ne("")]
    data["normalized"] = data["text"].map(normalize_text)
    conflicts = data.groupby("normalized")["intent"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError("Normalized phrases with conflicting labels: " + str(conflicts.index[:10].tolist()))
    return data.drop_duplicates(subset=["normalized"]).reset_index(drop=True)


def main():
    base = Path(__file__).resolve().parent
    data = load_dataset(base)
    X, y = data["normalized"], data["intent"]
    counts = y.value_counts().sort_index()
    if counts.min() < 5:
        raise ValueError("At least 5 examples per intent are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    holdout_model = make_model().fit(X_train, y_train)
    pred = holdout_model.predict(X_test)
    prob = holdout_model.predict_proba(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_test, prob, labels=holdout_model.classes_)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classification_report": classification_report(y_test, pred, zero_division=0),
    }
    print("Dataset rows:", len(data), "| intents:", y.nunique())
    print("Intent counts:\n", counts.to_string())
    print("\nStratified random holdout metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])
    # Fit the final artifact on all training rows after recording holdout results.
    final_model = make_model().fit(X, y)
    model_path = base / "voice_ubuntu_intent_model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    metadata = {
        "project": "Voice Ubuntu NLP", "version": "v6",
        "model_type": "TF-IDF word n-grams + character n-grams + Logistic Regression",
        "model_file": model_path.name, "dataset_file": "training_examples.csv",
        "training_examples": int(len(data)), "intent_count": int(y.nunique()),
        "intents": sorted(y.unique().tolist()),
        "intent_counts": {str(k): int(v) for k, v in counts.items()},
        "scikit_learn_version_at_build": __import__("sklearn").__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "evaluation_note": "Random split may be optimistic because templated utterances can be similar. Review independent_eval_v6.csv results too.",
        "safety": "This model predicts intent only. It never executes OS commands. Security validation is mandatory before operations.",
    }
    (base / "voice_ubuntu_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "holdout_metrics_v6.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved final model:", model_path)

if __name__ == "__main__":
    main()
