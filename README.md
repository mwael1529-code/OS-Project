# Voice Ubuntu NLP v6

## What changed in v6

Version 6 expands the training set from the original 315 examples to approximately 1,350 bilingual English/Arabic command examples across 15 intents. The dataset is more balanced across general Ubuntu actions and adds `PLAY_MEDIA` as a separate intent for the planned local music/movie feature.

The parser also improves several behaviors found during the v5 audit:

- recognizes common Egyptian Arabic requests to open YouTube, including `ممكن تفتح يوتيوب`;
- treats a bare `Play <title>` as local-media playback by default, while explicit YouTube requests are routed to YouTube search;
- extracts a local media `title`;
- extracts a `query` from file-search requests such as `Locate my project PDF`;
- extracts `folder_name` and optional `parent_folder` from folder-creation requests;
- asks for clarification when a folder/media target is missing.

These are NLP and parsing outputs only. The package does not execute operating-system actions.

## Intents

`CHANGE_VOLUME`, `CLOSE_APP`, `CREATE_FOLDER`, `GET_CPU`, `GET_DISK`, `GET_RAM`, `LOCK_SCREEN`, `OPEN_APP`, `OPEN_FOLDER`, `OPEN_YOUTUBE`, `PLAY_MEDIA`, `SEARCH_FILE`, `SEARCH_YOUTUBE`, `SHUTDOWN`, and `UNKNOWN`.

## Main files

- `voice_ubuntu_intent_model.joblib`: trained v6 classifier.
- `nlp_runtime.py`: text normalization and intent prediction.
- `voice_ubuntu_parser.py`: rule routing, multi-action splitting, and entity extraction.
- `training_examples.csv`: expanded bilingual training data used by the trainer.
- `training_examples_v5_original.csv`: unchanged baseline dataset for reproducibility.
- `training_examples_v6.csv`: expanded dataset copy.
- `prepare_training_data.py`: rebuilds the expanded dataset from the baseline CSV.
- `train_model.py`: measures a stratified random holdout, then fits and saves the release model.
- `independent_eval_v6.csv`: manually curated evaluation examples that are not part of the training set.
- `evaluate_model_v6.py`: evaluates the saved model against the held-out examples and writes a report.
- `test_voice_ubuntu_parser.py`, `test_nlp_runtime.py`, `test_voice_ubuntu_v6.py`: automated tests.
- `Voice_Ubuntu_NLP_v6_Colab_Train_Test_Export.ipynb`: Colab workflow to retrain, test, and export to Google Drive.
- `MODEL_CARD_V6.md`: model design, intended use, performance notes, and limitations.

## Run locally (Ubuntu or Windows)

Use Python 3.11+ and a virtual environment.

```bash
python -m venv .venv
```

On Ubuntu/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluate_model_v6.py
python demo_parser.py
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluate_model_v6.py
python demo_parser.py
```

To rebuild data and retrain:

```bash
python prepare_training_data.py
python train_model.py
python -m pytest -q
python evaluate_model_v6.py
```

## Use from another Python module

```python
from voice_ubuntu_parser import parse_text

result = parse_text("Search YouTube for Python tutorials")
print(result)
```

The parser returns intents and entities. The caller must pass these to the security component before any operating-system action. `Person 6` validates; `Person 5` executes approved operations. Never run raw recognized speech as a shell command and never use `shell=True` with spoken text.

## Example output for the local-media feature

Input: `Play Interstellar`

The parser should return `PLAY_MEDIA` with an entity like `{"title": "Interstellar"}`. It does not search the media catalogue or launch a media player; that is Person 5's responsibility.

Input: `Play Interstellar on YouTube`

The parser should return `SEARCH_YOUTUBE` and a YouTube search URL. The user still selects the result.

## Performance interpretation

The stratified random split can produce optimistic metrics because the dataset contains carefully generated phrase variants. The separate 40-row `independent_eval_v6.csv` is designed to help check unseen exact phrases and is checked for exact normalized overlap with the training set. However, it is still a small, manually curated test set—not independently collected real voice transcripts. Collect real English, Egyptian Arabic, and mixed-language examples before making production-quality accuracy claims.

## Safety and scope

The NLP module does not execute commands. The security module must independently apply allowlists, path validation, least privilege, confirmations for dangerous operations, and rejection handling before the Ubuntu action module is called. A parser test is not a full system security test.
