# Model Card — Voice Ubuntu NLP v6

## Summary

A lightweight offline intent classifier for a student voice-controlled Ubuntu assistant. It combines word-level TF-IDF n-grams and character-level TF-IDF n-grams with Logistic Regression. Explicit YouTube routing and entity extraction are implemented as parser rules around the classifier.

## Version changes

- Expanded the labeled data from the 315-row v5 baseline to a 1,348-row dataset.
- Balanced most command intents near 90 utterances per class, keeping more YouTube-search examples because the parser supports open-ended search topics.
- Added the `PLAY_MEDIA` intent to support the planned local music/movie catalogue feature.
- Added entity extraction for file search and folder creation, plus media title extraction.
- Improved the Egyptian Arabic pattern for opening YouTube.

## Intended use

Predict an intent and extract structured parameters from text that has already been transcribed by a speech-to-text component. Intended for an academic prototype, offline demo, and team integration tests.

## Not intended for

- direct execution of shell commands;
- unattended destructive or privileged operations;
- automatic playback before confirming that the local media title has been resolved;
- claims of reliable recognition across all dialects, accents, microphones, or noisy environments;
- use as a general-purpose assistant for requests outside the supported intent set.

## Training data

The expanded CSV contains 1,348 labeled phrases across 15 intents. It includes manually written bilingual examples and deterministic command-template variations. Most intents have 88–90 examples; `OPEN_YOUTUBE` has 83, and `SEARCH_YOUTUBE` has 98. This helps balance classes but does not replace a real user dataset. Some examples share command structures, so a random row-based split can overestimate generalization.

A baseline CSV and a separate `independent_eval_v6.csv` are included. The independent set has 40 hand-curated examples and no exact normalized phrase overlap with the training CSV. It is small and human-authored, so it should be treated as a smoke/generalization check rather than a definitive benchmark.

## Model design

- Text normalization: lowercase, whitespace normalization, remove Arabic diacritics/tatweel, normalize common alef variants.
- Word TF-IDF: unigrams and bigrams, up to 20,000 features.
- Character TF-IDF: `char_wb` n-grams of sizes 2–5, up to 35,000 features.
- Classifier: Logistic Regression, class-weight balancing, C=2.5, max_iter=4000, deterministic random state.
- Runtime output: intent, candidate intent, class probability score, margin, status, and source.

Probability scores are model outputs, not guarantees of correctness. Rule-routed YouTube outputs may use a score of 1.0 to denote a matched parser rule; that is not a calibrated probability.

## Validation performed

- Existing and new pytest suite: 33 tests passed in the development environment.
- Independent manually curated evaluation: tested separately against the model, with exact normalized overlaps removed.
- YouTube parser stress checks, malformed-input checks, media/file/folder entity smoke tests, and deterministic prediction checks.
- Stratified random holdout from the expanded training set produced approximately 98% accuracy in the local training run. Because of templated phrase variants, treat this as an optimistic internal estimate; do not compare it directly to external real-speech performance.

## Known limitations

- Test samples remain small and authored by the development team.
- ASR transcription errors, noise, accents, and speech latency are not covered by text-only tests.
- `SEARCH_FILE` query extraction identifies a search phrase; the Linux module must actually search the filesystem.
- `CREATE_FOLDER` extracts a requested name; the Linux module must validate the path and create the directory.
- `PLAY_MEDIA` extracts a title; the media module must resolve it to a local file and handle missing/ambiguous matches.
- Some natural phrasing may still be classified as `UNKNOWN` or the wrong intent.
- No end-to-end security certification is implied.

## Required safety boundary

Speech-to-text → NLP structured action → security validation/confirmation → Ubuntu action → response. The NLP component must never directly execute operating-system commands. Dangerous operations such as shutdown, process termination, and file deletion need explicit security policy and confirmation outside this model.
