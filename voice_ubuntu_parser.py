"""Text-to-structured-command parsing for Voice Ubuntu. No OS actions are run here."""
from pathlib import Path
from urllib.parse import urlencode
import configparser
import os
import re
import difflib
from nlp_runtime import predict_intent, normalize_text

YOUTUBE_HOME = "https://www.youtube.com/"

# Split only when the right-hand side looks like a new command. This avoids
# splitting searchable titles such as "rock and roll".
ACTION_CONNECTOR = re.compile(r"\s+(?:and then|then|and|وبعدين|وبعدها|ثم)\s+", re.IGNORECASE)
ACTION_START = re.compile(
    r"^(?:(?:please|can you|could you|would you|i want to|i'd like to)\s+)*"
    r"(?:open|launch|start|run|close|quit|exit|check|show|display|find|locate|create|make|set|increase|decrease|lower|raise|mute|lock|shut|turn off|search|look up|look for|watch|play|stream|go to)\b"
    r"|^(?:افتح|شغل|شغلي|شغللي|اقفل|أقفل|اغلق|اعرض|اظهر|دور|دورلي|ابحث|ابحثلي|هات|هاتلي|وريني|شوف|شوفلي|اعمل|انشئ|زود|ارفع|وطي|قلل|اكتم|اقفل|اطفي|اطفئ|عايز|عايزة|ممكن|ثم)",
    re.IGNORECASE,
)

APP_ALIASES = {
    "Firefox": ["Mozilla Firefox", "firefox", "فايرفوكس", "متصفح فايرفوكس"],
    "Visual Studio Code": ["VS Code", "Code", "في اس كود", "فيجوال ستوديو كود"],
    "Files": ["File Manager", "مدير الملفات", "الملفات"],
    "Terminal": ["ترمينال", "التيرمنال", "الطرفية"],
}

FOLDER_ALIASES = {
    "Downloads": ["التنزيلات", "الداونلودز", "مجلد التنزيلات"],
    "Documents": ["المستندات", "مجلد المستندات"],
    "Desktop": ["سطح المكتب"],
    "Music": ["الموسيقى", "مجلد الموسيقى"],
    "Videos": ["الفيديوهات", "مجلد الفيديوهات"],
    "Pictures": ["الصور", "مجلد الصور"],
}


def normalize_title(value):
    value = normalize_text(value)
    value = re.sub(r"[^\w\s؀-ۿ.-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def discover_ubuntu_applications():
    """Discover common XDG .desktop application entries. Run inside Ubuntu."""
    roots = []
    data_home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser()
    roots.append(data_home / "applications")
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    roots.extend(Path(p) / "applications" for p in data_dirs if p)
    records, seen = [], set()
    for root in roots:
        if not root.is_dir():
            continue
        for file in root.rglob("*.desktop"):
            try:
                parser = configparser.ConfigParser(interpolation=None, strict=False)
                parser.read(file, encoding="utf-8")
                if not parser.has_section("Desktop Entry"):
                    continue
                sec = parser["Desktop Entry"]
                if sec.get("Type", "Application") != "Application" or sec.getboolean("NoDisplay", fallback=False):
                    continue
                name = sec.get("Name", "").strip()
                if not name:
                    continue
                desktop_id = file.name
                key = (name.casefold(), desktop_id)
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "display_name": name, "desktop_id": desktop_id,
                    "desktop_file": str(file), "exec": sec.get("Exec", ""),
                    "try_exec": sec.get("TryExec", ""),
                })
            except (OSError, configparser.Error, ValueError):
                continue
    return records


def _application_names(app):
    display_name = str(app.get("display_name", ""))
    desktop_id = str(app.get("desktop_id", ""))
    names = [display_name, desktop_id, Path(desktop_id or "placeholder.desktop").stem]
    names.extend(APP_ALIASES.get(display_name, []))
    return [normalize_title(name) for name in names if name and normalize_title(name)]


def resolve_application(query, catalogue=None, ambiguity_margin=0.08):
    catalogue = discover_ubuntu_applications() if catalogue is None else catalogue
    if not catalogue:
        return {"status": "catalogue_unavailable", "query": query,
                "message": "No application catalogue found. Run discovery inside Ubuntu."}
    q = normalize_title(query)
    if not q:
        return {"status": "missing_target", "query": query}
    candidates = []
    for app in catalogue:
        names = _application_names(app)
        exact = any(q == name for name in names)
        contained = any(q in name or name in q for name in names)
        score = 1.0 if exact else (0.91 if contained else max(
            [difflib.SequenceMatcher(None, q, name).ratio() for name in names] or [0.0]
        ))
        if score >= 0.70:
            candidates.append((score, app))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return {"status": "not_found", "query": query}
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < ambiguity_margin:
        return {"status": "ambiguous", "query": query,
                "candidates": [item[1] for item in candidates[:5]]}
    score, best = candidates[0]
    return {"status": "found", "query": query, "display_name": best["display_name"],
            "desktop_id": best.get("desktop_id", ""), "score": round(score, 3),
            "desktop_file": best.get("desktop_file", "")}


def extract_application_query(text, catalogue=None):
    catalogue = discover_ubuntu_applications() if catalogue is None else catalogue
    normalized = normalize_title(text)
    names = []
    for app in catalogue or []:
        display_name = str(app.get("display_name", ""))
        names.append((normalize_title(display_name), display_name))
        names.extend((normalize_title(alias), display_name) for alias in APP_ALIASES.get(display_name, []))
    for alias, canonical in sorted(names, key=lambda pair: len(pair[0]), reverse=True):
        if alias and alias in normalized:
            return canonical
    q = normalize_text(text)
    patterns = [
        r"^please\s+", r"^could you\s+", r"^can you\s+", r"^would you\s+",
        r"^(?:open|launch|start|run|close|quit|exit)\s+",
        r"^(?:افتح|شغل|شغلي|اقفل|أقفل|اغلق|ممكن تفتح|عايز افتح|عايز اشغل)\s+",
        r"\s+(?:app|application|program|please|for me)$", r"\s+(?:التطبيق|البرنامج|ابلكيشن)$",
    ]
    for pattern in patterns:
        q = re.sub(pattern, "", q, count=1, flags=re.IGNORECASE).strip()
    return q.strip(" .,!؟،؛")


def resolve_user_folder(query):
    q = normalize_title(query)
    for name, aliases in FOLDER_ALIASES.items():
        if q in {normalize_title(name), *[normalize_title(alias) for alias in aliases]}:
            path = Path.home() / name
            if path.is_dir():
                return {"status": "found", "folder": name, "path": str(path)}
            return {"status": "not_found", "folder": name, "path": str(path)}
    return {"status": "not_found", "query": query}


# The rule layer handles common command shapes before the classifier. This makes
# arbitrary YouTube search topics work without listing the topic in training data.
_OPEN_YOUTUBE_PATTERNS = [
    re.compile(
        r"^(?:(?:please|can you|could you|would you|i want to|i'd like to|help me)\s+)*"
        r"(?:open|launch|go to|take me to|visit|bring up|navigate to|show me)\s+"
        r"(?:the\s+)?youtube(?:\s+(?:website|homepage|home page|home|page))?"
        r"(?:\s+in\s+(?:firefox|my browser|a web browser))?[.!?\s]*$", re.I),
    re.compile(
        r"^(?:(?:ممكن|عايز|عايزة|عاوز|عاوزه|عايزه|لو سمحت|من فضلك)\s+)*"
        r"(?:تفتح(?:لي)?|افتح(?:لي)?|ادخل على|روح على|روح ل|اذهب الى|وديني على|شغل موقع)\s+"
        r"(?:موقع\s+)?(?:ال)?يوتيوب"
        r"(?:\s+(?:على فايرفوكس|في المتصفح|الرئيسية|الصفحة الرئيسية))?[.!؟\s]*$", re.I),
]

# A command shape is sufficient to route a YouTube search, even when its topic
# was never present in the training examples. This is deliberate: the subject is
# free text, not a class that must be learned separately for each topic.
_SEARCH_START_PATTERNS = [
    # English explicit YouTube search commands.
    re.compile(
        r"^(?:(?:please|can you|could you|would you|i want to|i want you to|i need to|i'd like to|i'd like you to|help me)\s+)*"
        r"(?:open\s+(?:the\s+)?youtube\s+and\s+)?"
        r"(?:search(?:\s+(?:on|in)\s*)?\s+youtube|search\s+on\s+youtube|youtube\s+search|"
        r"search\s+for\s+.+\s+on\s+(?:the\s+)?youtube)\b", re.I),
    re.compile(
        r"^(?:(?:please|can you|could you|would you|i want to|i want you to|i need to|i'd like to|help me)\s+)*"
        r"(?:find|look\s+up|look\s+for|search)\b.*\b(?:on|in)\s+(?:the\s+)?youtube\b", re.I),
    # Search-oriented media requests; bare "play <title>" is reserved for local media.
    re.compile(
        r"^(?:(?:please|can you|could you|would you|i want to|i'd like to|help me)\s+)*"
        r"(?:find\s+(?:videos?|a\s+(?:song|video|movie|tutorial))|show\s+me\s+videos?|"
        r"search\s+for\s+(?:a\s+)?(?:song|video|movie|music)|look\s+for\s+videos?)\b", re.I),
    re.compile(
        r"^(?:(?:please|can you|could you|would you|i want to|i'd like to|help me)\s+)*"
        r"(?:watch|stream)\b.*\b(?:documentary|tutorial|lesson|lecture|videos?|how\s+to|movie|film)\b", re.I),
    # Arabic explicit search / media-finding requests.
    re.compile(
        r"^(?:(?:ممكن|عايز|عايزة|عاوز|عاوزه|عايزه)\s+)*"
        r"(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?"
        r"(?:ابحث|أبحث|إبحث|دور|دورلي|ادور|ادورلي|تدور|تدورلي)(?:\s+لي)?\s+"
        r"(?:على|في)\s+(?:ال)?يوتيوب\s+(?:على|عن)\b", re.I),
    re.compile(
        r"^(?:(?:ممكن|عايز|عايزة|عاوز|عاوزه|عايزه)\s+)*"
        r"(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?"
        r"(?:ابحث|أبحث|إبحث|ابحثلي|دور|دورلي|ادور|ادورلي|تدور|تدورلي)\s+عن\b", re.I),
    re.compile(
        r"^(?:(?:ممكن|عايز|عايزة|عاوز|عاوزه|عايزه)\s+)*"
        r"(?:عايز\s+فيديوهات\s+عن|عايزة\s+فيديوهات\s+عن|هاتلي\s+فيديو|هات\s+فيديو|"
        r"وريني\s+فيديو|دور\s+على\s+(?:اغنية|أغنية|فيديو|فيديوهات)|"
        r"ابحث\s+عن\s+(?:اغنية|أغنية|فيديو|فيديوهات)|شغل\s+فيديو\s+عن)\b", re.I),
]


def _clean_query_edges(query):
    """Trim only edge punctuation; preserve punctuation inside a search topic."""
    q = str(query or "").strip().strip("\"'“”‘’«»")
    q = re.sub(r"[.!?؟،؛]+$", "", q).strip()
    q = re.sub(r"^(?:for|about|عن|على|في)\s+", "", q, flags=re.I).strip()
    return q


def extract_youtube_query(text):
    """Extract arbitrary free-text from an English or Arabic YouTube command.

    The topic does not need to be known in advance or present in training data.
    This function strips the command frame, preserves the topic, and removes
    trailing platform words/punctuation before URL-encoding it.
    """
    q = str(text or "").strip()
    if not q:
        return ""

    # Iteratively remove polite request openers.
    polite_patterns = [
        r"^(?:please|hey|can you please|could you please|would you please|can you|could you|would you|i would like you to|i would like to|i want you to|i need you to|i want to|i need to|i'd like you to|i'd like to|help me)\s+",
        r"^(?:من فضلك|لو سمحت|ممكن|عايز|عايزة|عاوز|عاوزه|عايزه)\s+",
    ]
    for _ in range(5):
        old = q
        for pattern in polite_patterns:
            q = re.sub(pattern, "", q, count=1, flags=re.I).strip()
        if q == old:
            break

    # Longest and most specific command prefixes must be removed first.
    prefixes = [
        # English explicit platform framing.
        r"^open\s+(?:the\s+)?youtube\s+and\s+(?:search|look(?:\s+up)?|find)\s+(?:for\s+)?",
        r"^(?:search\s+(?:(?:on|in)\s+)?youtube|search\s+on\s+youtube|youtube\s+search|search\s+youtube)\s*(?:for|about)?\s*",
        r"^(?:find|look\s+for|look\s+up|search\s+for)\s+(?:videos?\s+(?:on\s+(?:the\s+)?youtube\s+)?about\s+|videos?\s+about\s+|me\s+)?",
        r"^(?:show\s+me\s+videos?\s+(?:on\s+(?:the\s+)?youtube\s+)?about|find\s+videos?\s+on\s+(?:the\s+)?youtube\s+about)\s+",
        r"^(?:watch|stream)\s+(?:a\s+)?(?:video\s+about\s+)?",
        r"^play\s+(?:(?:the|a|an)\s+)?(?:song|track|video|movie|music)\s+",
        r"^(?:search|find\s+me|find|look\s+up|look\s+for|show\s+me)\s+",
        r"^play\s+",
        # Arabic specific platform requests, including Egyptian variants.
        r"^(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?(?:ابحث|أبحث|إبحث|ابحثلي|ابحثلي)\s+(?:على|في)\s+(?:ال)?يوتيوب\s+(?:عن|على)\s*",
        r"^(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?(?:دور|دورلي|ادور|ادورلي|تدور|تدورلي)(?:لي)?\s+(?:على|في)\s+(?:ال)?يوتيوب\s+(?:عن|على)\s*",
        r"^(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?(?:ابحث|أبحث|إبحث|ابحثلي|دورلي|دور|ادور|ادورلي|تدور|تدورلي)\s+عن\s*",
        r"^(?:افتح\s+(?:ال)?يوتيوب\s*و\s*)?(?:ابحث|أبحث|إبحث)\s+(?:في|على)\s+(?:ال)?يوتيوب\s+عن\s*",
        r"^(?:دور|دورلي|ادور|ادورلي|تدور|تدورلي)\s+على\s+(?:ال)?يوتيوب\s+(?:على|عن)\s*",
        r"^(?:ابحث|أبحث|إبحث|ابحثلي|ابحثلي)\s+عن\s*",
        r"^(?:عايز|عايزة|عاوز|عاوزه|عايزه)\s+(?:اتفرج\s+على|اشوف|اسمع)\s*",
        r"^(?:عايز|عايزة|عاوز|عاوزه|عايزه)\s+فيديوهات\s+عن\s*",
        r"^(?:اتفرج\s+على|بتفرج\s+على|شغللي|شغلي|شغل|اسمعلي|اسمع|شوفلي|شوفي|شوف|هاتلي|هات|وريني|اعرض)\s*",
        r"^(?:فيديوهات|فيديو|مقاطع)\s+(?:عن|حول)\s*",
        r"^(?:اغنية|أغنية|الاغنية|الأغنية|song|music|video)\s+",
    ]
    for _ in range(4):
        old = q
        for pattern in prefixes:
            q = re.sub(pattern, "", q, count=1, flags=re.I).strip()
        if q == old:
            break

    # Remove the platform mention when it is command framing at the end.
    q = re.sub(r"\s+(?:on|in)\s+(?:the\s+)?youtube$", "", q, flags=re.I).strip()
    q = re.sub(r"\s+(?:على|في)\s+(?:ال)?يوتيوب$", "", q, flags=re.I).strip()
    return _clean_query_edges(q)


def build_youtube_search_url(query):
    query = str(query).strip()
    if not query:
        return ""
    return "https://www.youtube.com/results?" + urlencode({"search_query": query})


def detect_youtube_route(text):
    """Return OPEN_YOUTUBE / SEARCH_YOUTUBE when a known command pattern matches."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if any(pattern.match(raw) for pattern in _OPEN_YOUTUBE_PATTERNS):
        return "OPEN_YOUTUBE"
    normalized = normalize_text(raw)
    # A general search/lookup verb coupled with YouTube is an explicit search.
    youtube_word = bool(re.search(r"\byoutube\b|يوتيوب", normalized))
    search_verb = bool(re.search(
        r"\b(search|find|look up|look for|watch|play|stream|videos?|show me|what is|how to|why|explain|tutorial|lesson|documentary|lyrics|song|music)\b|"
        r"ابحث|دور|شغل|اسمع|شوف|هات|وريني|فيديوهات|اغنيه|اغنية|اعرض|شرح|طريقة|كيفية|تعلم|كورس|محاضرة|وثائقي", normalized, re.I
    ))
    if youtube_word and search_verb:
        return "SEARCH_YOUTUBE"
    # Play/watch requests are directed to YouTube in this project by design.
    if any(pattern.match(raw) for pattern in _SEARCH_START_PATTERNS):
        return "SEARCH_YOUTUBE"
    return None


def _starts_new_action(text):
    return bool(ACTION_START.match(text.strip()))


def split_actions(text):
    """Split compound commands only when the right segment looks like a new action.

    For example, don't split "Search YouTube for rock and roll" at "and".
    "Open YouTube and search for calculus" is kept as one YouTube search command.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    stripped = text.strip()
    if re.match(r"^(?:please\s+)?open\s+(?:the\s+)?youtube\s+and\s+search\b", stripped, re.I):
        return [stripped]
    if re.match(r"^(?:افتح\s+(?:ال)?يوتيوب\s+و(?:ابحث|دور)\b)", stripped, re.I):
        return [stripped]
    pieces = []
    start = 0
    for match in ACTION_CONNECTOR.finditer(stripped):
        right = stripped[match.end():].strip()
        left = stripped[start:match.start()].strip()
        if left and right and _starts_new_action(right):
            pieces.append(left.strip(" .,!؟،؛"))
            start = match.end()
    final = stripped[start:].strip(" .,!؟،؛")
    if final:
        pieces.append(final)
    return pieces or [stripped]



def _trim_command_punctuation(value):
    return re.sub(r"^[\s,.;:!?؟،؛]+|[\s,.;:!?؟،؛]+$", "", str(value or "")).strip()


def extract_file_query(text):
    """Extract a searchable file/folder phrase from a SEARCH_FILE command.

    Returns free text only; it never performs filesystem operations.
    """
    q = str(text or "").strip()
    if not q:
        return ""
    q = re.sub(
        r"^(?:(?:please|can you|could you|would you|help me|i want you to|i need you to)\s+)*",
        "", q, flags=re.I)
    q = re.sub(
        r"^(?:find me|find|locate|look for|look up|search for|search|show me|get me|where can i find|where is)\s+",
        "", q, flags=re.I)
    q = re.sub(
        r"^(?:من فضلك|لو سمحت|ممكن|عايز|عايزة|عاوز|عاوزه|عايزه)\s+", "", q, flags=re.I)
    q = re.sub(
        r"^(?:دورلي على|دور على|دورلي|دور|ابحثلي عن|ابحث عن|اعثر على|هاتلي|هات|فين|اين|أين)\s*", "", q, flags=re.I)
    q = re.sub(r"^(?:my|the|a|an)\s+", "", q, flags=re.I)
    q = re.sub(r"^(?:ملف|الملف|مستند|المستند|صورة|ملف ال)\s+", "", q)
    q = re.sub(r"\s+(?:file|document|please|for me)$", "", q, flags=re.I)
    q = re.sub(r"\s+(?:في|داخل)\s+(?:مجلد\s+)?(.+)$", lambda m: " " + m.group(0), q) if False else q
    return _trim_command_punctuation(q).strip("\"'“”‘’")


def extract_folder_name(text):
    """Extract proposed folder name; does not create the folder."""
    q = str(text or "").strip()
    if not q:
        return ""
    patterns = [
        (r"^(?:please\s+)?(?:create|make|add)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\s+in\s+(.+?)\s+(?:named|called)\s+(.+)$", True),
        (r"^(?:please\s+)?(?:create|make|add)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\s+(?:named|called)\s+(.+)$", False),
        (r"^(?:please\s+)?(?:create|make|add)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory)\s+(.+)$", False),
        (r"^(?:اعمل|أنشئ|انشئ|سوي|سوّي)\s+(?:لي\s+)?(?:فولدر|مجلد)(?:\s+(?:جديد|جديده))?\s+(?:اسمه|اسمها|باسم|اسم)\s+(.+)$", False),
        (r"^(?:اعمل|أنشئ|انشئ|سوي|سوّي)\s+(?:لي\s+)?(?:فولدر|مجلد)\s+(.+)$", False),
    ]
    for pattern, has_parent in patterns:
        match = re.match(pattern, q, flags=re.I)
        if not match:
            continue
        if has_parent:
            _parent, name = match.group(1), match.group(2)
        else:
            name = match.group(1)
        name = _trim_command_punctuation(name)
        name = re.sub(r"^(?:named|called|اسمه|اسمها|باسم|اسم)\s+", "", name, flags=re.I)
        name = re.sub(r"^(?:for)\s+", "", name, flags=re.I)
        name = re.sub(r"\s+(?:folder|directory)$", "", name, flags=re.I)
        return name.strip("\"'“”‘’")
    return ""


def extract_media_title(text):
    """Extract the requested local audio/video title without opening it."""
    q = str(text or "").strip()
    if not q:
        return ""
    q = re.sub(r"^(?:(?:please|can you|could you|would you|help me|i want to|i'd like to)\s+)*", "", q, flags=re.I)
    q = re.sub(r"^(?:play|start playing|play me)\s+", "", q, flags=re.I)
    q = re.sub(r"^(?:the\s+)?(?:song|track|movie|film|video|audio file|video file|file)\s+", "", q, flags=re.I)
    q = re.sub(r"^(?:شغللي|شغلني|شغلي|شغل|اسمعلي|اسمع|افتح)\s*", "", q, flags=re.I)
    q = re.sub(r"^(?:ال)?(?:اغنية|أغنية|فيلم|فيديو|مقطع|ملف صوتي|ملف فيديو)\s*", "", q, flags=re.I)
    q = re.sub(r"\s+(?:from my local library|from my music folder|from my movies folder|from my collection|saved on my computer|from Downloads|from the Music folder)$", "", q, flags=re.I)
    q = re.sub(r"\s+(?:المحفوظ عندي|الموجود عندي|من مجلد الموسيقى|من مجلد الأفلام|من مكتبتي|على الجهاز)$", "", q)
    q = _trim_command_punctuation(q)
    return q.strip("\"'“”‘’«»")

def parse_one_action(text, applications=None):
    route = detect_youtube_route(text)
    prediction = predict_intent(text)
    intent = route or prediction["intent"]
    source = "rule" if route else prediction.get("source", "model")
    result = {
        "text": text, "intent": intent,
        "candidate_intent": route or prediction.get("candidate_intent"),
        "score": 1.0 if route else prediction.get("score"),
        "margin": None if route else prediction.get("margin"),
        "source": source,
        "status": "ok" if route else prediction.get("status", "unknown"),
        "entities": {},
    }
    if result["status"] != "ok":
        return result

    if intent == "SEARCH_YOUTUBE":
        query = extract_youtube_query(text)
        generic_queries = {"", "a song", "the song", "a video", "the video", "anything", "something", "something to watch", "شيء", "حاجة"}
        if normalize_title(query) in {normalize_title(value) for value in generic_queries}:
            result.update(status="needs_clarification", message="What would you like me to search for on YouTube?")
            return result
        result["entities"].update({"query": query, "browser": "firefox", "url": build_youtube_search_url(query)})
        result["message"] = "Open Firefox to YouTube search results; the user selects the desired result."
    elif intent == "OPEN_YOUTUBE":
        result["entities"].update({"browser": "firefox", "url": YOUTUBE_HOME})
    elif intent in {"OPEN_APP", "CLOSE_APP"}:
        query = extract_application_query(text, applications)
        app = resolve_application(query, applications)
        result["entities"]["application_query"] = query
        result["application_resolution"] = app
        if app["status"] == "found":
            result["entities"].update({"application": app["display_name"], "desktop_id": app["desktop_id"]})
        else:
            result["status"] = "needs_clarification" if app["status"] == "ambiguous" else app["status"]
    elif intent == "PLAY_MEDIA":
        title = extract_media_title(text)
        if not title:
            result.update(
                status="needs_clarification",
                message="Which local music or movie should I play?",
            )
        else:
            result["entities"].update({"title": title})
            result["message"] = "Resolve this title against the local media library before playback."
    elif intent == "SEARCH_FILE":
        query = extract_file_query(text)
        if not query:
            result.update(
                status="needs_clarification",
                message="Which file should I search for?",
            )
        else:
            result["entities"]["query"] = query
            if re.search(r"\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|png|jpe?g|mp3|mp4|mkv|wav)\b", query, re.I):
                result["entities"]["filename"] = query
            location_match = re.search(r"\b(?:in|inside)\s+(Documents|Downloads|Desktop)\b", text, re.I)
            if location_match:
                result["entities"]["search_location"] = location_match.group(1)
    elif intent == "CREATE_FOLDER":
        folder_name = extract_folder_name(text)
        if not folder_name:
            result.update(
                status="needs_clarification",
                message="What should the new folder be called?",
            )
        else:
            result["entities"]["folder_name"] = folder_name
            parent_match = re.search(r"\b(?:in|inside)\s+(Documents|Downloads|Desktop)\b", text, re.I)
            if parent_match:
                result["entities"]["parent_folder"] = parent_match.group(1)
    elif intent == "OPEN_FOLDER":
        query = text.strip()
        for pattern in [r"^please\s+", r"^open\s+", r"^show\s+", r"^افتح\s+"]:
            query = re.sub(pattern, "", query, count=1, flags=re.I).strip()
        query = re.sub(r"\s+(?:folder|directory)$", "", query, flags=re.I).strip()
        query = re.sub(r"^(?:my|the)\s+", "", query, flags=re.I).strip()
        folder = resolve_user_folder(query)
        result["folder_resolution"] = folder
        if folder["status"] == "found":
            result["entities"].update({"folder": folder["folder"], "path": folder["path"]})
        else:
            result["status"] = folder["status"]
    return result


def parse_text(text, applications=None):
    parts = split_actions(text)
    if not parts:
        return {"raw_text": text, "actions": [], "status": "invalid_input"}
    actions = [parse_one_action(part, applications=applications) for part in parts]
    overall = "ok" if all(action.get("status") == "ok" for action in actions) else "needs_review"
    return {"raw_text": text, "actions": actions, "status": overall}
