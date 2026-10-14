from pathlib import Path
import json

from nlp_runtime import predict_intent, normalize_text


def test_model_file_exists():
    assert Path(__file__).with_name("voice_ubuntu_intent_model.joblib").is_file()


def test_model_metadata_is_v6_and_has_15_intents():
    p = Path(__file__).with_name("voice_ubuntu_model_metadata.json")
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert meta["version"] == "v6"
    assert meta["intent_count"] == 15
    assert "PLAY_MEDIA" in meta["intents"]


def test_invalid_empty_input_is_handled():
    assert predict_intent("")["status"] == "invalid_input"
    assert predict_intent(None)["status"] == "invalid_input"
    assert predict_intent(123)["status"] == "invalid_input"


def test_arabic_normalization_strips_diacritics_and_unifies_alif():
    assert normalize_text("إِفْتَحْ يوتيوب") == "افتح يوتيوب"


def test_core_intents_are_predicted():
    cases = {
        "Open Firefox": "OPEN_APP",
        "Close Firefox": "CLOSE_APP",
        "Check RAM": "GET_RAM",
        "Check CPU usage": "GET_CPU",
        "Locate my project PDF": "SEARCH_FILE",
        "Create a folder named Projects": "CREATE_FOLDER",
        "Play Interstellar": "PLAY_MEDIA",
        "Tell me a joke": "UNKNOWN",
    }
    for text, expected in cases.items():
        assert predict_intent(text)["intent"] == expected, text
