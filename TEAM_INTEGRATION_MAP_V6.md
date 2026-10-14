# Voice Ubuntu NLP v6 — Team Integration Map

## Pipeline

```text
Person 1: Microphone/audio capture
              |
              v
Person 2: Speech-to-text (transcribed text + detected language)
              |
              v
Person 3: NLP v6 (parse text -> structured actions/entities/status)
              |
              v
Person 6: Security policy + validation + confirmation
              |
              v
Person 5: Ubuntu action dispatcher / filesystem / app / media
              |
              +---------------------+
              |                     |
              v                     v
Person 4: TTS response       Person 7: GUI + history/logs
              \                     /
               \                   /
                v                 v
                 Person 8: integration + testing
```

## Responsibilities and direct integration points

### Person 1 — Audio and microphone

Produce a clean audio segment or audio file for Person 2. Do not implement intent classification here. Test quiet speech, room noise, pauses, clipping, and silence detection. This NLP package does not accept live audio; it accepts text.

### Person 2 — Speech-to-text

Transcribe audio and pass the recognized text into the NLP parser. Keep transcription concerns separate from command interpretation. A proposed minimal handoff is:

```python
transcript = "Search YouTube for Python tutorials"
result = parse_text(transcript, applications=application_catalogue)
```

If the STT module also returns `language` and `confidence`, keep those fields in the orchestrator's request context; the v6 parser currently takes the `text` and optional application catalogue.

### Person 3 — NLP / this package

Import `parse_text` from `voice_ubuntu_parser.py`. The parser returns:

- `raw_text`: original recognized text;
- `actions`: one or more structured actions;
- each action has `intent`, `candidate_intent`, `score`, `source`, `status`, and `entities`;
- top-level `status`: `ok`, `needs_review`, or `invalid_input`.

Example:

```python
from voice_ubuntu_parser import parse_text

result = parse_text("Create a folder in Documents called Notes")
# Expected action intent: CREATE_FOLDER
# Expected entities: {"folder_name": "Notes", "parent_folder": "Documents"}
```

v6 adds `PLAY_MEDIA` and the following entity conventions:

| Intent | Entity data produced by NLP v6 | What the action layer still needs to do |
|---|---|---|
| `OPEN_APP` / `CLOSE_APP` | `application_query`, and resolved `application` / `desktop_id` when a catalogue match is found | Confirm the action is allowed, then launch/close with safe APIs. |
| `OPEN_FOLDER` | `folder` and `path` when a known folder exists | Validate the resolved path and open it. |
| `SEARCH_FILE` | `query`; `filename` when a recognized extension is present; optional `search_location` | Search the filesystem within approved directories. |
| `CREATE_FOLDER` | `folder_name`; optional `parent_folder` | Validate name/path and create the directory. |
| `GET_CPU`, `GET_RAM`, `GET_DISK` | Intent only | Collect system metrics using trusted OS/system APIs. |
| `CHANGE_VOLUME` | Intent only | Parse/validate exact target level if required and adjust volume safely. |
| `LOCK_SCREEN`, `SHUTDOWN` | Intent only | Apply risk policy and user confirmation as required. |
| `OPEN_YOUTUBE` | `browser`, homepage `url` | Open the browser to the provided URL. |
| `SEARCH_YOUTUBE` | `query`, `browser`, encoded `url` | Open the URL; the user chooses the result. |
| `PLAY_MEDIA` | `title` | Search local media index; ask the user if zero or multiple matches occur; open a resolved path. |
| `UNKNOWN` or `needs_clarification` | Status/message where available | Ask a clarifying question or report unsupported command. |

### Person 4 — Text-to-speech and wake word

Receive the orchestrator's user-facing response. Speak success/failure messages, clarification prompts, and security confirmation requests. Do not speak classifier probabilities as if they were certainty. TTS should not block the event loop while audio processing is active.

### Person 5 — Ubuntu actions, files, applications, and media

Accept only actions approved by Person 6. Implement OS-side behavior for intents and entities. For `PLAY_MEDIA`, use `entities["title"]` to search a local music/movie index. NLP v6 does not scan media directories, resolve actual media paths, or launch a player.

For `SEARCH_FILE`, the returned `query` is a search phrase, not a guaranteed path. For `CREATE_FOLDER`, `folder_name` is proposed user input, not an approved path. Validate both before filesystem work.

For `SEARCH_YOUTUBE`, the parser provides a URL, but does not fetch or select a video. The described UX is to open search results and let the user select a result.

### Person 6 — Security and confirmation

Be a mandatory gate between NLP and actions. Validate intent allowlists, fields, paths, and arguments. Request explicit confirmation for destructive or disruptive operations, including shutdown, and block unexpected targets. Do not pass raw recognized speech to a shell. Do not use `shell=True` with text from the user or STT.

A suggested decision object (coordinate names with the integration owner) is:

```json
{
  "decision": "ALLOW | CONFIRM | BLOCK",
  "validated_action": {
    "intent": "CREATE_FOLDER",
    "entities": {"folder_name": "Notes", "parent_folder": "Documents"}
  }
}
```

This schema is illustrative; the package does not include Person 6's validator.

### Person 7 — GUI, history, and logs

Display original transcribed text, action intents, entity values, status, security decision, and the final result. Record timestamp, intent, approved target, action status, and cancellation/failure. Treat `score` as a model score, not a safety decision. Keep UI code separate from NLP, security, and Linux execution logic.

### Person 8 — Integration, tests, and deployment

Keep the shared interface stable; run unit tests per component and integration tests across every boundary. Integration order:

1. Fixed text -> NLP parser.
2. STT text -> NLP parser.
3. NLP action -> security decision.
4. Approved action -> Ubuntu action dispatcher.
5. Action result -> TTS and GUI/logging.
6. Full speech-to-result flow on the target Ubuntu desktop.

Run the tests in this package from the project folder:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluate_model_v6.py
```

## Mandatory boundary

**NLP predicts and parses; security authorizes; the action layer executes.** Passing NLP tests does not prove the security module, speech recognizer, or Ubuntu operations are correct. Test those components separately and together in the actual Ubuntu environment.
