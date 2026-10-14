"""Lightweight offline intent classification for the Voice Ubuntu project."""
from functools import lru_cache
from pathlib import Path
import re
import joblib

DEFAULT_MODEL_PATH = Path(__file__).with_name("voice_ubuntu_intent_model.joblib")


def normalize_text(text: str) -> str:
    """Conservative normalization shared by training and prediction."""
    text = str(text).lower().strip()
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)  # Arabic marks/tatweel
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"[،؛؟]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    return joblib.load(model_path)


def predict_intent(
    text: str,
    model_path=DEFAULT_MODEL_PATH,
    min_confidence: float = 0.28,
    min_margin: float = 0.05,
):
    """Predict a command intent; this function never executes an action.

    Scores are classifier probabilities, not guarantees of correctness.
    YouTube command routing is also handled by explicit rules in the parser
    so arbitrary query topics do not need to appear in the training set.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "intent": "UNKNOWN", "candidate_intent": None, "score": 0.0,
            "margin": 0.0, "status": "invalid_input", "source": "model",
        }
    model = _load_model(str(Path(model_path).resolve()))
    probabilities = model.predict_proba([normalize_text(text)])[0]
    order = probabilities.argsort()[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    candidate = str(model.classes_[best])
    score = float(probabilities[best])
    margin = float(probabilities[best] - probabilities[second])
    if candidate == "UNKNOWN":
        intent, status = "UNKNOWN", "unknown"
    elif score < min_confidence or margin < min_margin:
        intent, status = "UNKNOWN", "uncertain"
    else:
        intent, status = candidate, "ok"
    return {
        "intent": intent, "candidate_intent": candidate,
        "score": round(score, 4), "margin": round(margin, 4),
        "status": status, "source": "model",
    }


if __name__ == "__main__":
    print("Voice Ubuntu intent model. Type 'exit' to stop.")
    while True:
        text = input("Command: ").strip()
        if text.lower() == "exit":
            break
        print(predict_intent(text))
