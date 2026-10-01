# Changelog

## [Unreleased]

## [0.4.0] - 2026-10-01

### Changed
- **Breaking:** `ImageAnalysisTask` is built from a field roster, and its
  schema, prompt and parser are all derived from it. `ImageAnalysisTask::new()`
  now asks for `description` (one sentence of at most
  `ImageAnalysisTask::DEFAULT_DESCRIPTION_MAX_CHARS` = 120 characters) and
  `tags` (at most `ImageAnalysisTask::DEFAULT_TAGS_MAX_ITEMS` = 8 labels)
  only. Each of the other eight fields (`scene`, `subjects`, `objects`,
  `actions`, `emotion`, `shot_type`, `lighting`, `categories`) is an
  `Extension` switched on with `with_extensions`; a switched-off field is
  absent from the schema and the prompt, is refused as
  `JsonParseError::UnknownFields` if an answer carries it, and reads empty
  in the parsed `ImageAnalysis`. `with_extensions(Extension::ALL)` asks for
  all ten fields, as the task did before. `ImageAnalysis` keeps its ten
  fields.
- Both caps are stated in the JSON Schema (`maxLength` on `description`,
  `maxItems` on `tags`) and in the prompt, and `parse` refuses an answer over
  either cap as `JsonParseError::MissingFields` naming the field instead of
  truncating it. The `Grammar::JsonSchema` docs now list the keywords an
  engine serving `ImageAnalysisTask` must honour.
- **Breaking:** `ImageAnalysisTask::parse` accepts each field only in the
  JSON type its schema entry declares, so `Ok` means the answer satisfied the
  schema. The drift tolerances are gone: a comma-separated string for `tags`,
  a bare string for an array field (`subjects`, `objects`, `actions`,
  `emotion`, `lighting`, `categories`) and a one-element array for
  `shot_type` are each refused as `JsonParseError::MissingFields` naming the
  field, like any other value of an undeclared type. The answer must also be
  a JSON text: `parse` no longer trims whitespace JSON does not allow (a
  no-break space, a line separator) from around the object, and refuses it
  as `JsonParseError::Json`; JSON whitespace there still parses. A decoder
  that enforces the schema (the `Grammar::JsonSchema` contract) never
  produces any of those shapes. Leniency for an engine that does not
  enforce the grammar, if it is ever wanted, is a separate, explicit
  opt-in; none is built.
- `ImageAnalysisTask::parse` names the field whatever JSON value it holds.
  A number outside `f64`'s range (`1e400`) or a value nested deeper than
  serde_json's recursion limit of 128 used to fail the whole answer as
  `JsonParseError::Json` before its field could be named. In a field the
  task asks for, such a value is now `JsonParseError::MissingFields`
  naming that field; under any other key it is
  `JsonParseError::UnknownFields`; and a top-level value that is not an
  object names every asked-for field whatever it holds.
  `JsonParseError::Json` now means exactly that the answer is not a JSON
  text, and it comes first: an answer that is not JSON is `Json` even when
  it also repeats a key. A string escaping one half of a UTF-16 surrogate
  pair, which no Unicode text can hold, is `Json` wherever it sits. The
  `json` feature turns on serde_json's additive `raw_value` feature, which
  `parse` uses to check each field's JSON type before decoding its value.

### Added
- `image_analysis::Extension`, the closed set of the eight optional fields,
  with `ALL` and `as_str` (the field's JSON key).
- `ImageAnalysisTask::{has_extension, with_extensions, set_extensions}` and
  the cap accessors `description_max_chars` / `tags_max_items` with their
  `with_*` / `set_*` builders.

## [0.3.0] - 2026-08-31

### Added
- `ImageAnalysisTask` (`llmtask::image_analysis::ImageAnalysisTask`), behind
  the existing `json` feature: the canonical `Task` implementation for
  `ImageAnalysis` — prompt, JSON Schema, and a resilient parser — merged up
  from the two byte-for-byte-equivalent engine copies (`lfm/src/image_analysis.rs`
  and `qwen3-vl/src/image_analysis.rs`, both still pinned to `llmtask = "0.1"`
  and its pre-rename nine-field `ImageAnalysis`). Every engine now runs the
  same task instead of maintaining parallel copies; retiring the two
  downstream copies is each engine's own follow-up.
- Sealed label-prompt discipline: `IMAGE_ANALYSIS_PROMPT` now instructs, for
  every array field, lowercase/singular/1-3-word phrases with no trailing
  punctuation (previously the per-field word-count guidance varied field to
  field). The parser never lowercases at parse time — the discipline is
  prompt-only, pinned by `label_array_discipline_is_stated_in_prompt` and
  `parse_does_not_lowercase_labels`.
- `categories` joins the JSON Schema's `properties` as a **required** field
  (see `REQUIRED_FIELDS`), matching the other nine (`tags` included) — the
  newly-added `ImageAnalysis` field with no prior engine producing it.

### Changed
- `JsonParseError::MissingFields` now also names a listed field that's
  present with a JSON type its schema entry can't satisfy (e.g. a number
  where a string or array of strings is expected) — previously only
  absent/null required fields were named, and a wrong-type field fell
  through to a generic, unnamed `serde` error.
- `ImageAnalysisTask::parse` now fully enforces the schema it declares
  instead of only partially checking it: an object key outside the ten
  declared `properties` is rejected as a new `JsonParseError::UnknownFields`
  (the runtime counterpart of the schema's `additionalProperties: false`,
  which nothing previously checked against already-decoded JSON), and
  `categories` — now a required field, see above — is held to the exact
  same absent/null/wrong-type checks as the other nine, closing a gap where
  a present `categories: null` used to silently default to an empty list
  instead of being rejected.
- `ImageAnalysisTask::parse` now refuses a top-level JSON object that
  declares the same member name more than once, instead of silently
  keeping only the last copy. `serde_json`'s stock `Value` decode
  collapses a duplicate object member via last-write-wins `Map::insert`
  before any schema validation runs, so e.g. `{"categories": null,
  "categories": []}` used to parse successfully (the valid second copy
  quietly won), while the reverse key order was rejected — the same
  order-dependent bypass covered a wrong-typed duplicate of any required
  field, and a duplicated key could never be named by
  `JsonParseError::UnknownFields` once collapsed. The top-level object is
  now decoded through a duplicate-checking parse that refuses the first
  repeated member name it sees, before a `Value` is ever built, surfaced
  as a new `JsonParseError::DuplicateField` naming the key.

## [0.2.0] - 2026-08-31

### Added
- `ImageAnalysis::categories: Vec<SmolStr>` field, with `categories()` /
  `with_categories()` / `set_categories()` accessors — broad content-category
  labels, coarser-grained than `tags`.

### Changed
- **Breaking:** `ImageAnalysis::mood` renamed to `ImageAnalysis::emotion`
  (`mood()` / `with_mood()` / `set_mood()` → `emotion()` / `with_emotion()` /
  `set_emotion()`). `llmtask` is published at `0.1.0`, so this rename needs a
  pre-1.0 semver-breaking version bump on release. The downstream `lfm`
  engine's `ImageAnalysisTask` (prompt text, JSON schema, and
  `LfmScenePayload`, all in `lfm/src/image_analysis.rs`) still names the
  field `mood` throughout and needs a follow-up patch before `lfm` can pick
  up the new `llmtask` version.

## [0.1.0] - 2026-05-10

### Added
- Initial release: `Task` trait, `ParseError` enum, `ImageAnalysis` data type.
- `ImageAnalysis` accessor surface ported verbatim from `qwen::scene::SceneAnalysis`
  to host the canonical type once both `qwen` and `lfm` engines depend on this crate.
