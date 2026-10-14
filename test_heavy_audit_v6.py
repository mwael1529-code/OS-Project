"""Broader parser stress and deterministic-output checks for v6."""
import random
import string
from urllib.parse import parse_qs, urlparse

from nlp_runtime import predict_intent
from voice_ubuntu_parser import parse_text


def _youtube_topics():
    topics = [
        "Python tutorials", "rock and roll music", "C++ pointers and references",
        "Linux; permissions & ownership", "100% CPU usage",
        "machine learning: from basics to projects", "شرح التفاضل والتكامل",
        "تحليل البيانات باستخدام Excel", "مقارنة بين SQL و Python",
        "Python و SQL للمبتدئين", "الذكاء الاصطناعي والاحصاء",
        "café & crème brûlée", "résumé writing tips", "question & answer",
        "foo/bar", "quoted string with 'internal' quotes", "emoji 🤖 Python",
        "Unicode Ελληνικά", "space science 2026", "SQL joins + indexes",
        "how to use pandas groupby", "file search and folder management",
        "history of Cairo", "التعلم المراقب وغير المراقب",
        "topic with #tag and @mentions", "words with brackets [like this]",
        "Python: beginner to intermediate", "statistical inference and probability",
        "C# and .NET fundamentals", "explain Bayes theorem",
        "شرح pandas groupby بالعربي", "المتوسط والوسيط والمنوال",
        "تنصيب لينكس على virtualbox", "الفرق بين correlation and causation",
    ]
    topics.extend(f"topic {i}: C++ and Python / version {i}" for i in range(40))
    return topics


def test_74_youtube_topics_survive_entity_extraction_and_url_encoding():
    topics = _youtube_topics()
    assert len(topics) == 74  # 34 curated topics + 40 generated topics
    for topic in topics:
        result = parse_text(f"Search YouTube for {topic}", applications=[])
        action = result["actions"][0]
        assert action["intent"] == "SEARCH_YOUTUBE", topic
        assert action["status"] == "ok", topic
        assert action["entities"]["query"] == topic, topic
        decoded = parse_qs(urlparse(action["entities"]["url"]).query)["search_query"]
        assert decoded == [topic], topic


def test_100_randomized_inputs_do_not_break_parser_contract():
    rng = random.Random(20261012)
    alphabet = string.ascii_letters + string.digits + " .,!?;:-_/#&()[]+% " + "ابتثجحخدذرزسشصضطظعغفقكلمنهوي" + "éΩ"
    for _ in range(100):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        result = parse_text(text, applications=[])
        assert isinstance(result, dict)
        assert "status" in result
        assert "actions" in result
        assert isinstance(result["actions"], list)


def test_common_classifier_predictions_are_repeatable():
    cases = [
        "Open Firefox", "Close Firefox", "Check RAM", "Check CPU usage",
        "Search YouTube for Python tutorials", "ابحث على يوتيوب عن شرح التفاضل",
        "Play Interstellar", "Locate my project PDF", "Tell me a joke",
    ]
    for text in cases:
        outputs = [predict_intent(text) for _ in range(5)]
        signatures = {
            (item.get("intent"), item.get("candidate_intent"), item.get("score"), item.get("status"))
            for item in outputs
        }
        assert len(signatures) == 1, text
