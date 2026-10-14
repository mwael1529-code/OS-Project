"""Offline regression tests for Voice Ubuntu NLP v5."""
from voice_ubuntu_parser import parse_text, extract_youtube_query, build_youtube_search_url, split_actions

QUERY_CASES = {
    "Search YouTube for Python tutorials": "Python tutorials",
    "Search for quantum physics on YouTube": "quantum physics",
    "Open YouTube and search for calculus": "calculus",
    "Search YouTube for rock and roll music": "rock and roll music",
    "Search in YouTube for Python": "Python",
    "YouTube search for C++ pointers": "C++ pointers",
    "What is linear algebra on YouTube": "What is linear algebra",
    "ابحث على يوتيوب عن شرح التفاضل": "شرح التفاضل",
    "ابحث في يوتيوب عن تحليل البيانات": "تحليل البيانات",
    "افتح يوتيوب وابحث عن اساسيات بايثون": "اساسيات بايثون",
    "افتح يوتيوب و ابحث عن اساسيات بايثون": "اساسيات بايثون",
    "دور على يوتيوب على اغاني ام كلثوم": "اغاني ام كلثوم",
    "شغل أغنية تملي معاك": "تملي معاك",
    "عايز فيديوهات عن تحليل البيانات": "تحليل البيانات",
    "عايز اتفرج على وثائقي عن الفضاء": "وثائقي عن الفضاء",
    "I would like you to search YouTube for a video about the causes of inflation": "a video about the causes of inflation",
    "Search YouTube for how to install Ubuntu in VirtualBox": "how to install Ubuntu in VirtualBox",
    "Can you search YouTube for C++ pointers and references?": "C++ pointers and references",
    "Find videos about Pandas on YouTube": "Pandas",
    "دورلي على يوتيوب على ازاي اتعلم بايثون": "ازاي اتعلم بايثون",
    "عايز ادور على يوتيوب على تحليل البيانات": "تحليل البيانات",
    "ابحث عن شرح الاحتمالات على اليوتيوب": "شرح الاحتمالات",
    "ممكن تدورلي على يوتيوب على نظرية بايز": "نظرية بايز",
}


def test_free_text_query_extraction():
    for text, expected in QUERY_CASES.items():
        assert extract_youtube_query(text) == expected


def test_search_term_with_and_is_not_split():
    assert split_actions("Search YouTube for rock and roll music") == ["Search YouTube for rock and roll music"]


def test_compound_action_splits_correctly():
    assert split_actions("Open Firefox and search YouTube for rock and roll") == [
        "Open Firefox", "search YouTube for rock and roll"
    ]


def test_arabic_url_is_encoded():
    url = build_youtube_search_url("شرح التفاضل")
    assert url.startswith("https://www.youtube.com/results?")
    assert "search_query=" in url
    assert "شرح التفاضل" not in url  # Arabic is percent encoded safely


def test_open_youtube_homepage():
    result = parse_text("Open YouTube")
    assert result["actions"][0]["intent"] == "OPEN_YOUTUBE"
    assert result["actions"][0]["entities"]["url"] == "https://www.youtube.com/"


def test_missing_query_requests_clarification():
    result = parse_text("Search YouTube for")
    assert result["actions"][0]["status"] == "needs_clarification"


def test_compound_command_with_mock_application_catalogue():
    mock_apps = [{
        "display_name": "Firefox",
        "desktop_id": "firefox.desktop",
        "desktop_file": "/mock/firefox.desktop",
    }]
    result = parse_text("Open Firefox and search YouTube for rock and roll", applications=mock_apps)
    assert result["status"] == "ok"
    assert [action["intent"] for action in result["actions"]] == ["OPEN_APP", "SEARCH_YOUTUBE"]
    assert result["actions"][1]["entities"]["query"] == "rock and roll"



def test_arbitrary_unseen_topics_are_routed_to_youtube():
    examples = [
        ("Search YouTube for astrophysics lecture 2026", "astrophysics lecture 2026"),
        ("ابحث على يوتيوب عن شرح خوارزمية ديكسترا", "شرح خوارزمية ديكسترا"),
        ("Search YouTube for a recipe with chickpeas and tahini", "a recipe with chickpeas and tahini"),
    ]
    for command, expected_query in examples:
        result = parse_text(command)
        action = result["actions"][0]
        assert action["intent"] == "SEARCH_YOUTUBE"
        assert action["status"] == "ok"
        assert action["entities"]["query"] == expected_query


def test_url_encodes_arbitrary_bilingual_query():
    from urllib.parse import parse_qs, urlparse
    query = "شرح C++ و pointers"
    url = build_youtube_search_url(query)
    assert parse_qs(urlparse(url).query)["search_query"] == [query]

def test_regular_file_search_is_not_routed_to_youtube():
    result = parse_text("Find my report file")
    assert result["actions"][0]["intent"] == "SEARCH_FILE"
    result = parse_text("Search for notes.txt")
    assert result["actions"][0]["intent"] == "SEARCH_FILE"
