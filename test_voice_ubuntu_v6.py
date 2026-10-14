from urllib.parse import parse_qs, urlparse

from voice_ubuntu_parser import (
    parse_text,
    extract_file_query,
    extract_folder_name,
    extract_media_title,
    detect_youtube_route,
    build_youtube_search_url,
)

MOCK_APPS = [{
    "display_name": "Firefox",
    "desktop_id": "firefox.desktop",
    "desktop_file": "/mock/firefox.desktop",
}]


def action(result, index=0):
    return result["actions"][index]


def test_egyptian_arabic_open_youtube_route():
    result = parse_text("ممكن تفتح يوتيوب")
    assert action(result)["intent"] == "OPEN_YOUTUBE"
    assert action(result)["entities"]["url"] == "https://www.youtube.com/"


def test_local_media_defaults_to_play_media():
    result = parse_text("Play Interstellar")
    assert action(result)["intent"] == "PLAY_MEDIA"
    assert action(result)["entities"]["title"] == "Interstellar"


def test_arabic_local_song_title_is_extracted():
    result = parse_text("شغل أغنية تملي معاك")
    assert action(result)["intent"] == "PLAY_MEDIA"
    assert action(result)["entities"]["title"] == "تملي معاك"


def test_explicit_youtube_play_request_routes_to_search():
    result = parse_text("Play Interstellar on YouTube")
    assert action(result)["intent"] == "SEARCH_YOUTUBE"
    assert action(result)["entities"]["query"] == "Interstellar"


def test_media_collection_suffix_is_removed():
    assert extract_media_title("Play the movie Interstellar from my collection") == "Interstellar"


def test_file_search_query_extraction():
    assert extract_file_query("Locate my project PDF") == "project PDF"
    assert extract_file_query("Find my report file") == "report"


def test_file_search_adds_query_entity():
    result = parse_text("Locate my project PDF")
    a = action(result)
    assert a["intent"] == "SEARCH_FILE"
    assert a["entities"]["query"] == "project PDF"


def test_folder_name_entity_extraction():
    result = parse_text("Create a folder called Semester 4")
    assert action(result)["entities"]["folder_name"] == "Semester 4"


def test_folder_parent_entity_extraction():
    result = parse_text("Create a folder in Documents called Notes")
    a = action(result)
    assert a["entities"]["folder_name"] == "Notes"
    assert a["entities"]["parent_folder"] == "Documents"


def test_arabic_folder_name_extraction():
    result = parse_text("اعمل مجلد اسمه الجامعة")
    assert action(result)["entities"]["folder_name"] == "الجامعة"


def test_missing_folder_name_requires_clarification():
    result = parse_text("Create a new folder")
    assert result["status"] == "needs_review"
    assert action(result)["status"] == "needs_clarification"


def test_file_query_unsupported_route_does_not_become_youtube():
    assert detect_youtube_route("Locate my project PDF") is None
    result = parse_text("Locate my project PDF")
    assert action(result)["intent"] == "SEARCH_FILE"


def test_youtube_query_keeps_and_inside_topic():
    result = parse_text("Search YouTube for rock and roll")
    a = action(result)
    assert a["entities"]["query"] == "rock and roll"
    assert parse_qs(urlparse(a["entities"]["url"]).query)["search_query"] == ["rock and roll"]


def test_arabic_youtube_url_round_trip():
    phrase = "شرح الاحتمال الشرطي"
    url = build_youtube_search_url(phrase)
    assert parse_qs(urlparse(url).query)["search_query"] == [phrase]


def test_compound_command_splits_without_breaking_query():
    result = parse_text("Open Firefox and search YouTube for rock and roll", applications=MOCK_APPS)
    assert [a["intent"] for a in result["actions"]] == ["OPEN_APP", "SEARCH_YOUTUBE"]
    assert result["actions"][1]["entities"]["query"] == "rock and roll"


def test_unsupported_request_is_flagged():
    result = parse_text("Tell me a joke")
    assert action(result)["intent"] == "UNKNOWN"
    assert result["status"] == "needs_review"


def test_empty_input_is_handled_without_exception():
    assert parse_text("")["status"] == "invalid_input"
    assert parse_text("   ")["status"] == "invalid_input"


def test_app_resolution_remains_a_parser_not_executor():
    result = parse_text("Open Firefox", applications=MOCK_APPS)
    assert action(result)["entities"]["desktop_id"] == "firefox.desktop"
    assert "executed" not in result
    assert "executed" not in action(result)
