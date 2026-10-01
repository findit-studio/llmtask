//! `ImageAnalysis` — the canonical single-image VLM output type — and
//! [`ImageAnalysisTask`] (behind the `json` feature) — the canonical
//! `Task` implementation that produces it: prompt, JSON Schema, and a
//! parser that holds the answer to that schema.
//!
//! Both live here so every engine (`lfm`, `qwen3-vl`, and future
//! consumers such as `mediagraph`'s VLM node) runs the exact same
//! task instead of maintaining parallel copies. Before this module
//! absorbed it, `lfm` and `qwen3-vl` each carried their own
//! `ImageAnalysisTask` — byte-for-byte equivalent down to the
//! parser's resilience rules, both still pinned to `llmtask = "0.1"`
//! and its pre-rename nine-field `ImageAnalysis` (`mood`, no
//! `categories`). Retiring those local copies in favor of this one is
//! each engine's own follow-up, not done here.
//!
//! `ImageAnalysis` itself only needs `alloc`; `ImageAnalysisTask`
//! additionally needs `serde_json::Value` (the schema) and
//! [`crate::JsonParseError`], so it's gated on `json`.
//!
//! The type is named for what it holds (analysis of an image) rather
//! than the upstream use case (representing a video scene via a
//! keyframe). The `scene` field still carries the scene-category
//! label within the analysis.

use smol_str::SmolStr;
// Bring `Vec` into scope under both std (resolves via the
// `extern crate std`) and alloc-only (resolves via the
// `extern crate alloc as std` alias in lib.rs).
use std::vec::Vec;

/// Structured single-image VLM output. Construct via an engine's
/// `ImageAnalysisTask::parse` (the `Task::parse` impl) or, for
/// tests/builders, [`ImageAnalysis::new`]
/// followed by `with_*` chains. All fields are private; the accessor
/// surface follows the rest of the crate's `scenesdetect`-style getter /
/// `with_*` / `set_*` convention.
///
/// Detection-array fields (`subjects` / `objects` / `actions` / `emotion` /
/// `lighting`) are `Vec<SmolStr>` — flat label lists, no per-detection
/// confidence. Wrapping each label in a `Detection { label, confidence }`
/// would require a confidence source the VLM can't reliably provide —
/// VLM self-reported confidence is poorly calibrated, and a hardcoded
/// placeholder is a no-op for both UX and search-time ranking. If a
/// downstream consumer needs per-detection scoring, the practical
/// sources are search-time embedding similarity or scene-aggregation
/// metrics, not VLM self-report.
///
/// A field the producing task did not ask for reads empty: the canonical
/// `ImageAnalysisTask` asks for `description` and `tags` by default, and
/// for each other field only when that field's extension is switched on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageAnalysis {
  scene: SmolStr,
  description: SmolStr,
  subjects: Vec<SmolStr>,
  objects: Vec<SmolStr>,
  actions: Vec<SmolStr>,
  emotion: Vec<SmolStr>,
  shot_type: SmolStr,
  lighting: Vec<SmolStr>,
  tags: Vec<SmolStr>,
  categories: Vec<SmolStr>,
}

impl ImageAnalysis {
  /// Construct an empty `ImageAnalysis` (all fields default).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new() -> Self {
    Self::default()
  }

  // --- scene (SmolStr: empty = absent) ---

  /// Short scene category (e.g. `"office"`, `"airport arrivals hall"`).
  /// Returns the empty string when the model didn't classify the scene
  /// (SCENE_PROMPT instructs the model to use empty strings for unknown
  /// fields). Check `scene().is_empty()` to test for absence.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn scene(&self) -> &str {
    &self.scene
  }

  /// Builder-style setter for `scene`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_scene(mut self, val: impl Into<SmolStr>) -> Self {
    self.scene = val.into();
    self
  }

  /// In-place setter for `scene`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_scene(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.scene = val.into();
    self
  }

  // --- description ---

  /// Free-form scene description (one length-capped sentence from the
  /// canonical `ImageAnalysisTask`), or empty when the model produced no
  /// description (e.g., on a low-information frame).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn description(&self) -> &str {
    &self.description
  }

  /// Builder-style setter for `description`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_description(mut self, val: impl Into<SmolStr>) -> Self {
    self.description = val.into();
    self
  }

  /// In-place setter for `description`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_description(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.description = val.into();
    self
  }

  // --- subjects ---

  /// Distinct people or animals visible in the scene.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn subjects(&self) -> &[SmolStr] {
    &self.subjects
  }

  /// Builder-style setter for `subjects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_subjects(mut self, val: Vec<SmolStr>) -> Self {
    self.subjects = val;
    self
  }

  /// In-place setter for `subjects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_subjects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.subjects = val;
    self
  }

  // --- objects ---

  /// Notable, search-relevant objects.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn objects(&self) -> &[SmolStr] {
    &self.objects
  }

  /// Builder-style setter for `objects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_objects(mut self, val: Vec<SmolStr>) -> Self {
    self.objects = val;
    self
  }

  /// In-place setter for `objects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_objects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.objects = val;
    self
  }

  // --- actions ---

  /// Visible actions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn actions(&self) -> &[SmolStr] {
    &self.actions
  }

  /// Builder-style setter for `actions`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_actions(mut self, val: Vec<SmolStr>) -> Self {
    self.actions = val;
    self
  }

  /// In-place setter for `actions`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_actions(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.actions = val;
    self
  }

  // --- emotion ---

  /// Scene-level emotion terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn emotion(&self) -> &[SmolStr] {
    &self.emotion
  }

  /// Builder-style setter for `emotion`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_emotion(mut self, val: Vec<SmolStr>) -> Self {
    self.emotion = val;
    self
  }

  /// In-place setter for `emotion`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_emotion(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.emotion = val;
    self
  }

  // --- shot_type ---

  /// One short camera-shot label (e.g. `"wide shot"`, `"close-up"`),
  /// or empty when the model didn't pick one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn shot_type(&self) -> &str {
    &self.shot_type
  }

  /// Builder-style setter for `shot_type`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_shot_type(mut self, val: impl Into<SmolStr>) -> Self {
    self.shot_type = val.into();
    self
  }

  /// In-place setter for `shot_type`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_shot_type(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.shot_type = val.into();
    self
  }

  // --- lighting ---

  /// Lighting terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn lighting(&self) -> &[SmolStr] {
    &self.lighting
  }

  /// Builder-style setter for `lighting`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_lighting(mut self, val: Vec<SmolStr>) -> Self {
    self.lighting = val;
    self
  }

  /// In-place setter for `lighting`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_lighting(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.lighting = val;
    self
  }

  // --- tags ---

  /// Short English search tags in lowercase (a count-capped list from the
  /// canonical `ImageAnalysisTask`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn tags(&self) -> &[SmolStr] {
    &self.tags
  }

  /// Builder-style setter for `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_tags(mut self, val: Vec<SmolStr>) -> Self {
    self.tags = val;
    self
  }

  /// In-place setter for `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_tags(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.tags = val;
    self
  }

  // --- categories ---

  /// Broad content-category labels, coarser-grained than `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn categories(&self) -> &[SmolStr] {
    &self.categories
  }

  /// Builder-style setter for `categories`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_categories(mut self, val: Vec<SmolStr>) -> Self {
    self.categories = val;
    self
  }

  /// In-place setter for `categories`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_categories(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.categories = val;
    self
  }
}

// Tests run under both std (default) and `--no-default-features
// --features alloc`: `vec!` / `format!` are alloc macros, not
// std-only — bring them into scope explicitly so the std prelude
// isn't required.
#[cfg(test)]
mod tests {
  use super::*;
  use std::vec;

  #[test]
  fn default_is_empty() {
    let s = ImageAnalysis::new();
    assert!(s.scene().is_empty());
    assert!(s.description().is_empty());
    assert!(s.subjects().is_empty());
    assert_eq!(s, ImageAnalysis::default());
  }

  #[test]
  fn builder_chains() {
    let s = ImageAnalysis::new()
      .with_scene("airport")
      .with_description("travelers walking through terminal")
      .with_subjects(vec!["middle-aged woman".into(), "child".into()])
      .with_emotion(vec!["busy".into()])
      .with_tags(vec!["airport".into(), "travel".into(), "indoor".into()])
      .with_categories(vec!["travel".into()]);
    assert_eq!(s.scene(), "airport");
    assert_eq!(s.subjects().len(), 2);
    assert_eq!(s.emotion().len(), 1);
    assert_eq!(s.tags().len(), 3);
    assert_eq!(s.categories().len(), 1);
  }

  #[test]
  fn set_in_place() {
    let mut s = ImageAnalysis::new();
    s.set_scene("plaza");
    s.set_emotion(vec!["calm".into()]);
    s.set_categories(vec!["landscape".into()]);
    assert_eq!(s.scene(), "plaza");
    assert_eq!(s.emotion().len(), 1);
    assert_eq!(s.categories().len(), 1);
  }
}

// ===== ImageAnalysisTask (json feature only) =====

#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
pub use image_analysis_task::{Extension, ImageAnalysisTask};

/// The `ImageAnalysisTask` implementation. Grouped in its own private
/// module (mirroring `task::json`'s pattern) so the whole surface —
/// struct, impls, the prompt/schema consts, and the parser's helper
/// functions — shares one `#[cfg(feature = "json")]` gate on the `mod`
/// line instead of repeating it on every item.
#[cfg(feature = "json")]
mod image_analysis_task {
  use core::{
    cell::RefCell,
    fmt::{self, Write as _},
    num::NonZeroUsize,
  };

  use serde::de::{self, Deserializer as _, MapAccess, Visitor};
  use serde_json::{Map, Value, json, value::RawValue};
  use smol_str::SmolStr;
  // Heap types under both std (resolves via the `extern crate std`) and
  // alloc-only (resolves via the `extern crate alloc as std` alias in
  // lib.rs).
  use std::{collections::BTreeMap, string::String, vec::Vec};

  use super::ImageAnalysis;
  use crate::{
    grammar::Grammar,
    task::{JsonParseError, Task},
  };

  /// The prompt's opening, before one paragraph per field the task asks
  /// for (see `ImageAnalysisTask::build_prompt`).
  const PROMPT_HEAD: &str = r#"Analyze the following video keyframes (in chronological order) from a single scene.

Return ONLY a valid JSON object with exactly these fields:
"#;

  /// The prompt's closing rules, after the field paragraphs. They hold for
  /// every roster: the label discipline speaks to whichever array fields
  /// the task asks for, and `tags` is always one of them.
  const PROMPT_RULES: &str = r#"
Rules:
- Use only information supported by the keyframes.
- Prefer concrete visual facts over speculation.
- Every array field's elements are lowercase, singular, 1-3-word phrases, with no trailing punctuation.
- Keep arrays deduplicated.
- Use empty arrays or empty strings when a field is unknown.
- Do not return markdown or any text outside the JSON object."#;

  /// The ten fields of the image-analysis JSON contract, in
  /// [`ImageAnalysis`] field order: the order a task's schema requires
  /// them in, its prompt describes them in, and its `parse` names them in.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  enum Field {
    Scene,
    Description,
    Subjects,
    Objects,
    Actions,
    Emotion,
    ShotType,
    Lighting,
    Tags,
    Categories,
  }

  impl Field {
    const ALL: [Self; 10] = [
      Self::Scene,
      Self::Description,
      Self::Subjects,
      Self::Objects,
      Self::Actions,
      Self::Emotion,
      Self::ShotType,
      Self::Lighting,
      Self::Tags,
      Self::Categories,
    ];

    /// The JSON key the contract spells for this field.
    const fn key(self) -> &'static str {
      match self {
        Self::Scene => "scene",
        Self::Description => "description",
        Self::Subjects => "subjects",
        Self::Objects => "objects",
        Self::Actions => "actions",
        Self::Emotion => "emotion",
        Self::ShotType => "shot_type",
        Self::Lighting => "lighting",
        Self::Tags => "tags",
        Self::Categories => "categories",
      }
    }

    /// The JSON type this field's schema entry declares, which is also the
    /// only shape `parse` accepts for it.
    const fn shape(self) -> Shape {
      match self {
        Self::Scene | Self::Description | Self::ShotType => Shape::String,
        Self::Subjects
        | Self::Objects
        | Self::Actions
        | Self::Emotion
        | Self::Lighting
        | Self::Tags
        | Self::Categories => Shape::StringArray,
      }
    }
  }

  /// A field's JSON type. [`ImageAnalysisTask::build_schema`] declares it
  /// and [`decode_field`] reads it and nothing else, both taking it from
  /// [`Field::shape`], so `parse` cannot accept a shape the schema does not
  /// declare.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  enum Shape {
    /// `{"type": "string"}`.
    String,
    /// `{"type": "array", "items": {"type": "string"}}`.
    StringArray,
  }

  /// One of the eight optional fields of the image-analysis contract. A
  /// task asks for an extension's field only when
  /// [`ImageAnalysisTask::with_extensions`] switches it on.
  ///
  /// `description` and `tags` are not extensions: every task asks for
  /// them, because they are the indexable content search consumes.
  /// Switching an extension on adds its field to the task's schema
  /// (`properties` and `required`) and its paragraph to the prompt, and
  /// makes `parse` require and read the field. While it is off, the field
  /// is in none of the three: an answer that carries it anyway is refused
  /// as [`JsonParseError::UnknownFields`], and the parsed [`ImageAnalysis`]
  /// reads it as empty.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum Extension {
    /// `scene`: one short scene-category label.
    Scene,
    /// `subjects`: the distinct people or animals visible.
    Subjects,
    /// `objects`: notable, search-relevant objects.
    Objects,
    /// `actions`: visible actions.
    Actions,
    /// `emotion`: descriptors of the scene's overall emotional tone.
    Emotion,
    /// `shot_type`: one camera-shot label.
    ShotType,
    /// `lighting`: lighting descriptors.
    Lighting,
    /// `categories`: broad content categories, coarser-grained than `tags`.
    Categories,
  }

  // `ImageAnalysisTask` keeps its extensions as one bit each in a `u8`.
  const _: () = assert!(Extension::ALL.len() <= u8::BITS as usize);

  impl Extension {
    /// Every extension, in [`ImageAnalysis`] field order.
    /// `ImageAnalysisTask::new().with_extensions(Extension::ALL)` asks for
    /// all ten fields.
    pub const ALL: [Self; 8] = [
      Self::Scene,
      Self::Subjects,
      Self::Objects,
      Self::Actions,
      Self::Emotion,
      Self::ShotType,
      Self::Lighting,
      Self::Categories,
    ];

    /// The field's JSON key: the name the contract spells, which is also
    /// the name of the field's [`ImageAnalysis`] accessor.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn as_str(&self) -> &'static str {
      self.field().key()
    }

    /// The contract field this extension switches on.
    const fn field(self) -> Field {
      match self {
        Self::Scene => Field::Scene,
        Self::Subjects => Field::Subjects,
        Self::Objects => Field::Objects,
        Self::Actions => Field::Actions,
        Self::Emotion => Field::Emotion,
        Self::ShotType => Field::ShotType,
        Self::Lighting => Field::Lighting,
        Self::Categories => Field::Categories,
      }
    }

    /// This extension's bit in `ImageAnalysisTask`'s extension mask.
    const fn bit(self) -> u8 {
      1 << (self as u8)
    }
  }

  /// The image-analysis task: a prompt, a JSON Schema and a parser, all
  /// derived from one FIELD ROSTER.
  ///
  /// The roster always holds `description` — one sentence of at most
  /// [`description_max_chars`](Self::description_max_chars) characters —
  /// and `tags` — at most [`tags_max_items`](Self::tags_max_items) labels.
  /// [`ImageAnalysisTask::new`] asks for those two and nothing else; each
  /// of the other eight fields is an [`Extension`] a deployment switches on
  /// with [`with_extensions`](Self::with_extensions). The schema declares
  /// and requires exactly the roster's fields, the prompt describes exactly
  /// those, and [`parse`](Task::parse) requires exactly those and leaves
  /// every other field of the returned [`ImageAnalysis`] empty.
  ///
  /// Both caps are stated in the schema (`maxLength` on `description`,
  /// `maxItems` on `tags`) and in the prompt. The constrained decoder
  /// enforces the schema, caps included, and `parse` holds the answer to
  /// the same schema: the keyword contract is on [`Grammar::JsonSchema`],
  /// and a field that breaks it is refused as
  /// [`JsonParseError::MissingFields`] naming the field. `parse` never
  /// truncates or rewrites the model's words to make them fit.
  ///
  /// # Example
  ///
  /// ```
  /// use llmtask::{
  ///   JsonParseError, Task,
  ///   image_analysis::{Extension, ImageAnalysisTask},
  /// };
  ///
  /// // The default task asks for one capped description sentence and one
  /// // capped label array.
  /// let task = ImageAnalysisTask::new();
  /// assert!(!task.prompt().is_empty());
  /// assert!(task.grammar().is_json_schema());
  ///
  /// let raw = r#"{
  ///   "description": "Two people talk across a desk in a bright office.",
  ///   "tags": ["office", "meeting"]
  /// }"#;
  /// let analysis = task.parse(raw).expect("parse should succeed");
  /// assert_eq!(analysis.tags().len(), 2);
  /// // A field the task did not ask for reads empty.
  /// assert!(analysis.scene().is_empty());
  ///
  /// // A deployment switches on the extensions it wants; each one joins the
  /// // schema, the prompt and the parse.
  /// let task = ImageAnalysisTask::new().with_extensions([Extension::Scene, Extension::ShotType]);
  /// let raw = r#"{
  ///   "scene": "office", "description": "Two people talk across a desk.",
  ///   "shot_type": "wide shot", "tags": ["office", "meeting"]
  /// }"#;
  /// let analysis = task.parse(raw).expect("parse should succeed");
  /// assert_eq!(analysis.scene(), "office");
  /// assert_eq!(analysis.shot_type(), "wide shot");
  ///
  /// // Each field must have the JSON type the schema declares for it.
  /// // `tags` is an array of strings, so one comma-separated string is
  /// // refused by name rather than split.
  /// let raw = r#"{"description": "Two people talk.", "tags": "office, meeting"}"#;
  /// assert!(matches!(
  ///   ImageAnalysisTask::new().parse(raw),
  ///   Err(JsonParseError::MissingFields(fields)) if fields == ["tags"]
  /// ));
  ///
  /// // Only an answer that is not JSON is `JsonParseError::Json`. A number
  /// // too large for an `f64` is still JSON, so in a string field it is
  /// // refused by the field's name.
  /// let raw = r#"{"description": 1e400, "tags": ["office"]}"#;
  /// assert!(matches!(
  ///   ImageAnalysisTask::new().parse(raw),
  ///   Err(JsonParseError::MissingFields(fields)) if fields == ["description"]
  /// ));
  /// ```
  #[derive(Clone)]
  pub struct ImageAnalysisTask {
    // One `Extension::bit` per switched-on extension.
    extensions: u8,
    description_max_chars: NonZeroUsize,
    tags_max_items: NonZeroUsize,
    accept_empty: bool,
    // Both derived from `extensions` and the two caps by `rebuild`, which
    // every setter of those three calls, so the schema, the prompt and
    // `parse` always describe the same roster.
    schema: Value,
    prompt: String,
  }

  impl ImageAnalysisTask {
    /// The default [`description_max_chars`](Self::description_max_chars):
    /// room for one sentence that says who is present, what they are doing,
    /// the setting and the mood. The constrained decoder closes the string
    /// at the cap, so the cap sits above a typical one-sentence caption
    /// rather than at it.
    pub const DEFAULT_DESCRIPTION_MAX_CHARS: NonZeroUsize = NonZeroUsize::new(120).unwrap();

    /// The default [`tags_max_items`](Self::tags_max_items): enough labels
    /// for keyword search to find a picture by its subject, setting and
    /// style, while keeping the answer short.
    pub const DEFAULT_TAGS_MAX_ITEMS: NonZeroUsize = NonZeroUsize::new(8).unwrap();

    /// Construct the default task: `description` and `tags` only, capped
    /// at [`Self::DEFAULT_DESCRIPTION_MAX_CHARS`] characters and
    /// [`Self::DEFAULT_TAGS_MAX_ITEMS`] tags, with no extension switched
    /// on and `accept_empty = false` (a payload that lacks the required
    /// indexable content is treated as a model regression and rejected;
    /// see [`Self::with_accept_empty`] for the full predicate and the
    /// opt-in alternative).
    pub fn new() -> Self {
      let mut task = Self {
        extensions: 0,
        description_max_chars: Self::DEFAULT_DESCRIPTION_MAX_CHARS,
        tags_max_items: Self::DEFAULT_TAGS_MAX_ITEMS,
        accept_empty: false,
        schema: Value::Null,
        prompt: String::new(),
      };
      task.rebuild();
      task
    }

    // --- extensions ---

    /// Returns whether the task asks for `extension`'s field.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn has_extension(&self, extension: Extension) -> bool {
      (self.extensions & extension.bit()) != 0
    }

    /// Builder-style setter for the extensions: the task asks for
    /// `description`, `tags` and exactly the fields of `extensions` (an
    /// extension listed twice counts once), replacing whatever extensions
    /// were on before. The schema and the prompt are rebuilt to match.
    pub fn with_extensions(mut self, extensions: impl IntoIterator<Item = Extension>) -> Self {
      self.set_extensions(extensions);
      self
    }

    /// In-place setter for the extensions. See [`Self::with_extensions`].
    pub fn set_extensions(&mut self, extensions: impl IntoIterator<Item = Extension>) -> &mut Self {
      self.extensions = extensions
        .into_iter()
        .fold(0, |mask, extension| mask | extension.bit());
      self.rebuild();
      self
    }

    // --- caps ---

    /// The most characters `description` may hold, counted in Unicode
    /// scalar values (what JSON Schema's `maxLength` counts). Stated in the
    /// schema as `maxLength` and in the prompt; `parse` refuses a longer
    /// description by name.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn description_max_chars(&self) -> NonZeroUsize {
      self.description_max_chars
    }

    /// Builder-style setter for `description_max_chars`. The schema and the
    /// prompt are rebuilt to match.
    pub fn with_description_max_chars(mut self, val: NonZeroUsize) -> Self {
      self.set_description_max_chars(val);
      self
    }

    /// In-place setter for `description_max_chars`. The schema and the
    /// prompt are rebuilt to match.
    pub fn set_description_max_chars(&mut self, val: NonZeroUsize) -> &mut Self {
      self.description_max_chars = val;
      self.rebuild();
      self
    }

    /// The most labels `tags` may hold. Stated in the schema as `maxItems`
    /// and in the prompt; `parse` refuses an answer listing more by name.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn tags_max_items(&self) -> NonZeroUsize {
      self.tags_max_items
    }

    /// Builder-style setter for `tags_max_items`. The schema and the prompt
    /// are rebuilt to match.
    pub fn with_tags_max_items(mut self, val: NonZeroUsize) -> Self {
      self.set_tags_max_items(val);
      self
    }

    /// In-place setter for `tags_max_items`. The schema and the prompt are
    /// rebuilt to match.
    pub fn set_tags_max_items(&mut self, val: NonZeroUsize) -> &mut Self {
      self.tags_max_items = val;
      self.rebuild();
      self
    }

    // --- accept_empty ---

    /// Returns whether the parser accepts payloads that lack the
    /// required indexable content (`description` AND `tags` both
    /// non-empty). See [`Self::with_accept_empty`] for the trade-off.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn accept_empty(&self) -> bool {
      self.accept_empty
    }

    /// Builder-style setter for `accept_empty`.
    ///
    /// When `false` (default), the parser rejects payloads that lack
    /// the required indexable content as [`JsonParseError::NoUsableFields`].
    /// The composite threshold accepts a payload when **either**:
    ///
    /// - `description` AND `tags` are both populated (the prose +
    ///   keyword path), OR
    /// - at least one of the **substantive** detection buckets —
    ///   `subjects`, `objects`, or `actions` — is non-empty (the
    ///   substantive-detection path; preserves who/what/where search
    ///   metadata even when the model fails to summarize). Only a task
    ///   that switches those extensions on can take this path; for the
    ///   default task the threshold is `description` AND `tags`.
    ///
    /// Style/attribute buckets (`emotion`, `lighting`, `categories`)
    /// and single-label fields (`scene`, `shot_type`) are intentionally
    /// NOT in the substantive path. A payload like `lighting: ["natural
    /// light"]` or `emotion: ["calm"]` alone (description and tags
    /// empty, no substantive detections) is more often a regression
    /// than a legitimate weak-but-real scene; rejecting it surfaces the
    /// failure instead of writing a single-attribute stub to the search
    /// index.
    ///
    /// When `true`, the parser bypasses the indexable-content check and
    /// returns whatever round-trips through the schema. The prompt's rules
    /// tell the model to "Use empty arrays or empty strings when a field
    /// is unknown", so on truly low-information frames (blank,
    /// fade-to-black, plain color) compliant model output can legitimately
    /// be sparse or fully-empty. Turn it on if your pipeline distinguishes
    /// "low-information scene" from "no useful content" via something
    /// other than the parser.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn with_accept_empty(mut self, val: bool) -> Self {
      self.accept_empty = val;
      self
    }

    /// In-place setter for `accept_empty`. See
    /// [`Self::with_accept_empty`] for the trade-off.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn set_accept_empty(&mut self, val: bool) -> &mut Self {
      self.accept_empty = val;
      self
    }
  }

  // ===== derivation from the roster =====
  impl ImageAnalysisTask {
    /// Re-derives the schema and the prompt from the extensions and the
    /// caps.
    fn rebuild(&mut self) {
      self.schema = self.build_schema();
      self.prompt = self.build_prompt();
    }

    /// Returns whether the task asks for `field`: `description` and `tags`
    /// always, any other field while its extension is switched on.
    fn asks_for(&self, field: Field) -> bool {
      Extension::ALL
        .into_iter()
        .find(|extension| extension.field() == field)
        .is_none_or(|extension| self.has_extension(extension))
    }

    /// The fields the task asks for, in [`ImageAnalysis`] field order.
    fn fields(&self) -> impl Iterator<Item = Field> {
      Field::ALL
        .into_iter()
        .filter(move |&field| self.asks_for(field))
    }

    /// JSON Schema for the task: every field it asks for is in
    /// `properties` and in `required`, and no other field is in either.
    /// `additionalProperties: false` declares that any field outside this
    /// list is invalid — a switched-off extension's field included — a
    /// constraint a constrained-decoding engine can enforce at generation
    /// time, but `parse` cannot inherit for free: reading already-generated
    /// text accepts any key regardless of what the schema says, so `parse`
    /// enforces this promise itself via `unknown_fields`, and each field's
    /// type and cap via `usable_value`.
    fn build_schema(&self) -> Value {
      let properties: Map<String, Value> = self
        .fields()
        .map(|field| (String::from(field.key()), self.property_schema(field)))
        .collect();
      let required: Vec<&str> = self.fields().map(Field::key).collect();
      json!({
          "type": "object",
          "properties": properties,
          "required": required,
          "additionalProperties": false
      })
    }

    /// `field`'s entry in the schema's `properties`: the JSON type of its
    /// [`Shape`], plus the task's cap on `description` and on `tags`.
    fn property_schema(&self, field: Field) -> Value {
      let mut property = match field.shape() {
        Shape::String => json!({ "type": "string" }),
        Shape::StringArray => json!({ "type": "array", "items": { "type": "string" } }),
      };
      match field {
        Field::Description => property["maxLength"] = self.description_max_chars.get().into(),
        Field::Tags => property["maxItems"] = self.tags_max_items.get().into(),
        Field::Scene
        | Field::Subjects
        | Field::Objects
        | Field::Actions
        | Field::Emotion
        | Field::ShotType
        | Field::Lighting
        | Field::Categories => {}
      }
      property
    }

    /// The prompt for the task: [`PROMPT_HEAD`], one paragraph per field
    /// the task asks for (see [`Self::write_paragraph`]), then
    /// [`PROMPT_RULES`].
    ///
    /// The wording is ported (structure and resilience-relevant wording)
    /// from the two engine-side copies this task replaced —
    /// `lfm/src/image_analysis.rs` and `qwen3-vl/src/image_analysis.rs`,
    /// themselves a verbatim port of the legacy `findit-qwen` service's
    /// prompt — with `mood` renamed to `emotion`, a `categories` field
    /// added, and the per-field word-count guidance collapsed into one
    /// discipline rule applied uniformly to every array field (see
    /// [`PROMPT_RULES`]).
    ///
    /// Intentionally written WITHOUT enumerated example values. Both of
    /// today's `llmtask` consumers (`lfm`, `qwen3-vl`) run on mistralrs,
    /// whose deterministic/greedy sampling applies a `presence_penalty`
    /// over the full context (prompt + generated tokens) — any
    /// value-token the prompt enumerates as an example (e.g. "office",
    /// "wide shot", "birthday cake with candles") gets a negative logit
    /// shift before the model emits anything, biasing generation away
    /// from that exact term even when a scene legitimately matches it.
    /// `llmtask` itself has no engine dependency and doesn't assume this
    /// mechanism is universal, but avoiding enumerated examples costs
    /// nothing and is free insurance for any engine that behaves this
    /// way — so format guidance stays in descriptive constraints (word
    /// counts, lowercase, singular) instead of `e.g. "..."` examples.
    fn build_prompt(&self) -> String {
      let mut prompt = String::from(PROMPT_HEAD);
      for field in self.fields() {
        // Writing into a `String` cannot fail.
        let _ = self.write_paragraph(&mut prompt, field);
        prompt.push('\n');
      }
      prompt.push_str(PROMPT_RULES);
      prompt
    }

    /// Writes `field`'s paragraph of the prompt. An extension's paragraph
    /// is the same whichever roster holds it. `description` and `tags`
    /// state the task's caps: the description is one sentence of at most
    /// `description_max_chars` characters, and `tags` lists at most
    /// `tags_max_items` labels.
    fn write_paragraph(&self, prompt: &mut String, field: Field) -> fmt::Result {
      match field {
        Field::Scene => prompt.write_str(
          "scene: a single short scene-category label in lowercase English, 1-3 words, no full sentence.",
        ),
        Field::Description => write!(
          prompt,
          "description: one concise sentence in English, at most {} characters, describing the stable visual facts across the scene. Cover who is present, what they are doing, the setting, and the overall mood or visual style. If readable on-screen text appears, quote that text first, then continue the description.",
          self.description_max_chars
        ),
        Field::Subjects => prompt.write_str(
          "subjects: array of distinct people or animals with visible distinguishing features.",
        ),
        Field::Objects => prompt.write_str("objects: array of notable, search-relevant objects."),
        Field::Actions => prompt.write_str("actions: array of visible actions."),
        Field::Emotion => prompt
          .write_str("emotion: array of descriptors for the scene's overall emotional tone."),
        Field::ShotType => prompt.write_str(
          "shot_type: a single short camera-shot label in lowercase English, 1-2 words (a cinematography term).",
        ),
        Field::Lighting => prompt.write_str("lighting: array of lighting descriptors."),
        Field::Tags => write!(
          prompt,
          "tags: array of at most {} short English search tags. Prefer high-confidence search terms, complementary synonyms, style words, and culture-specific terms only when visually supported.",
          self.tags_max_items
        ),
        Field::Categories => prompt
          .write_str("categories: array of broad content categories, coarser-grained than tags."),
      }
    }

    /// Names the keys of `members` that the task's schema does not declare
    /// — any key outside the fields the task asks for, a switched-off
    /// extension's field included. The runtime enforcement of the schema's
    /// `additionalProperties: false` (see [`Self::build_schema`]).
    ///
    /// Owned [`SmolStr`] rather than `&'static str`: unlike a declared
    /// field name, an unknown key isn't known at compile time — it's
    /// whatever text the decoder emitted.
    fn unknown_fields(&self, members: &BTreeMap<String, &RawValue>) -> Vec<SmolStr> {
      members
        .keys()
        .filter(|key| !self.fields().any(|field| field.key() == key.as_str()))
        .map(SmolStr::new)
        .collect()
    }

    /// `field`'s value in the answer, `value`, decoded as the task can use
    /// it: present, of exactly the JSON type the schema declares for the
    /// field (see [`decode_field`]), and within any cap the schema declares
    /// (see [`Self::exceeds_cap`]). `Ok(None)` leaves the field unusable:
    /// absent, `null`, of any other JSON type, or over a cap. Keys outside
    /// the roster are a separate concern, handled by
    /// [`Self::unknown_fields`].
    ///
    /// Folding "wrong type" into the same named-field list as
    /// "missing"/"null" is a deliberate hardening over the two engine
    /// copies this task replaces: those only pre-checked null/absent,
    /// then let a wrong-type field fall through to `serde`'s untagged-enum
    /// deserialization, which fails with a generic "data did not match
    /// any variant" message that never names the offending field. A value
    /// over a cap is folded in for the same reason: the schema promises
    /// the cap, so such a value has drifted from the schema as a wrong-typed
    /// one has, and `parse` refuses it by name rather than truncating the
    /// model's words to fit. `parse` names every unusable field at once.
    fn usable_value(
      &self,
      field: Field,
      value: Option<&RawValue>,
    ) -> Result<Option<FieldValue>, serde_json::Error> {
      let Some(value) = value else {
        return Ok(None);
      };
      Ok(decode_field(field, value)?.filter(|value| !self.exceeds_cap(field, value)))
    }

    /// `true` iff `value` holds more than the cap `field`'s schema entry
    /// declares, counted the way the schema counts: `description` in
    /// Unicode scalar values of the string as the answer wrote it (before
    /// `parse` trims it), `tags` in elements of the array.
    fn exceeds_cap(&self, field: Field, value: &FieldValue) -> bool {
      match (field, value) {
        (Field::Description, FieldValue::String(description)) => {
          description.chars().count() > self.description_max_chars.get()
        }
        (Field::Tags, FieldValue::Strings(tags)) => tags.len() > self.tags_max_items.get(),
        _ => false,
      }
    }
  }

  impl Default for ImageAnalysisTask {
    fn default() -> Self {
      Self::new()
    }
  }

  impl Task for ImageAnalysisTask {
    type Output = ImageAnalysis;
    type Value = Value;
    type ParseError = JsonParseError;

    fn prompt(&self) -> &str {
      &self.prompt
    }

    fn schema(&self) -> &Value {
      &self.schema
    }

    fn grammar(&self) -> Grammar {
      // Clone the cached JSON Schema once per call. Cheap relative to
      // constraint compilation, and matches the `schema()` contract.
      Grammar::JsonSchema(self.schema.clone())
    }

    fn parse(&self, raw: &str) -> Result<Self::Output, JsonParseError> {
      // Not a plain `serde_json::from_str` into a `Value` (see
      // `parse_members`'s doc comment): that collapses a duplicate
      // top-level member — later overwrites earlier — before any check
      // below ever sees both copies, and it fails on valid JSON (a number
      // outside `f64`'s range, deep nesting) before the field holding it
      // can be named. `raw` goes in untrimmed: the deserializer skips the
      // JSON whitespace a JSON text may carry around its value, and any
      // other character there is not JSON.
      let Some(members) = parse_members(raw)? else {
        // Not a JSON object at all: by definition every field the task
        // asks for is absent. Naming them via `MissingFields` is more
        // informative than a generic "expected top-level object" error,
        // and needs nothing beyond `serde_json` to construct.
        return Err(JsonParseError::MissingFields(
          self.fields().map(Field::key).collect(),
        ));
      };
      // Runtime enforcement of `additionalProperties: false` (see
      // `build_schema`): the schema declaring it constrains a
      // constrained-decoding engine, but nothing about reading the
      // already-generated text — which accepts any key — enforces it.
      // Checked before the declared fields so an object with both an
      // unknown key and a missing/invalid declared field is named for the
      // unknown key first (a structural violation of "which keys are even
      // allowed" takes precedence over per-field shape checks).
      let unknown = self.unknown_fields(&members);
      if !unknown.is_empty() {
        return Err(JsonParseError::UnknownFields(unknown));
      }
      let mut values = Vec::new();
      let mut unusable = Vec::new();
      for field in self.fields() {
        match self.usable_value(field, members.get(field.key()).copied())? {
          Some(value) => values.push((field, value)),
          None => unusable.push(field.key()),
        }
      }
      if !unusable.is_empty() {
        return Err(JsonParseError::MissingFields(unusable));
      }
      // Every field now holds exactly the shape its schema entry declares,
      // so the extraction cannot fail. A field the task does not ask for
      // is never read and keeps its empty default.
      let mut result = ImageAnalysis::new();
      for (field, value) in values {
        match field {
          Field::Scene => result.set_scene(value.into_label()),
          Field::Description => result.set_description(value.into_label()),
          Field::Subjects => result.set_subjects(value.into_labels()),
          Field::Objects => result.set_objects(value.into_labels()),
          Field::Actions => result.set_actions(value.into_labels()),
          Field::Emotion => result.set_emotion(value.into_labels()),
          Field::ShotType => result.set_shot_type(value.into_label()),
          Field::Lighting => result.set_lighting(value.into_labels()),
          Field::Tags => result.set_tags(value.into_labels()),
          Field::Categories => result.set_categories(value.into_labels()),
        };
      }
      // Indexable-content gate. The prompt's rules instruct the model to
      // "Use empty arrays or empty strings when a field is unknown",
      // so a truly compliant response on a blank/fade-to-black frame can
      // be partially or fully empty. But a decoder/model regression on a
      // normal frame also produces sparse output, and silently
      // overwriting real search metadata with that is worse than
      // failing. See `with_accept_empty` for the full predicate.
      if !self.accept_empty && lacks_indexable_content(&result) {
        return Err(JsonParseError::NoUsableFields);
      }
      Ok(result)
    }
  }

  // ===== reading the answer =====

  /// Reads `raw` as one JSON text and returns the members of its top-level
  /// object, each key decoded and each value kept as the JSON text the
  /// answer wrote it in ([`RawValue`]), or `None` when the top-level value
  /// is not an object.
  ///
  /// No value is decoded here. The field checks decode a declared field's
  /// value only once its first byte shows the JSON type the schema
  /// declares (see [`decode_field`]). Decoding every value into a
  /// [`Value`] up front, as `serde_json::from_str` does, fails on valid
  /// JSON before the field holding it can be named: a number outside
  /// `f64`'s range (`1e400`) is `NumberOutOfRange`, and nesting deeper than
  /// serde_json's limit of 128 is `RecursionLimitExceeded`.
  ///
  /// `raw` is read in two passes:
  ///
  /// 1. serde_json checks that `raw` is a JSON text, one value with nothing
  ///    but JSON whitespace around it, and captures that value as a
  ///    [`RawValue`]. The capture checks the grammar without converting
  ///    anything: a number is read digit by digit and never becomes an
  ///    `f64`, so its magnitude cannot fail it, and nested arrays and
  ///    objects are walked with a stack on the heap rather than by
  ///    recursion, so their depth cannot fail it either. A string's escapes
  ///    are checked for form but not decoded, which lets through the one
  ///    escape the grammar admits and no Unicode text can hold: half a
  ///    UTF-16 surrogate pair (`\uD800` alone). Decoding such a string
  ///    fails, so [`find_lone_surrogate`] refuses the answer as not JSON
  ///    wherever the escape sits — in a key, a field, an undeclared member
  ///    or a nested value — not only where a string gets decoded.
  /// 2. For an object, [`MembersVisitor`] reads the members, decoding each
  ///    key and capturing each value, and refuses a member name that
  ///    appears twice.
  ///
  /// So [`JsonParseError::Json`] means exactly that `raw` is not a JSON
  /// text, and it comes before every other refusal, a duplicate key
  /// included.
  ///
  /// The members are read by [`MembersVisitor`] rather than a stock map
  /// decode because a stock decode builds the object by repeated
  /// insertion: a later member overwrites an earlier one with no trace left
  /// behind. `{"categories": null, "categories": []}` would read as
  /// `{"categories": []}`, and the reverse key order as
  /// `{"categories": null}` alone. Both orderings are schema violations no
  /// compliant decoder should emit, but only the second would be caught —
  /// by the field check, and only because `null` happened to survive the
  /// collapse — and a duplicated key **outside** the declared fields could
  /// never be named by `unknown_fields` at all, because by then only one
  /// copy is left. While the members are read is the only point where
  /// both copies are still visible to compare.
  ///
  /// The duplicate check covers the top level only, deliberately: every
  /// property a task's schema can declare is `string` or `array of string`
  /// — this contract has no `object`-valued field, at the top level or
  /// nested. A JSON object can therefore only legitimately appear as the
  /// document root; anywhere else (e.g. an object in place of a string in
  /// a `subjects` array) it is already a wrong-shaped value that
  /// [`decode_field`] refuses on its first byte, whatever its own keys. A
  /// future field that legitimately nests an object would need this same
  /// duplicate check extended to it.
  ///
  /// This is the only place `parse` reads `raw` itself: the field checks
  /// decode only text captured here, and nothing strips a fence or retries
  /// a lenient parse. `reject_fenced_json` and
  /// `reject_json_with_wrapper_text` (in the tests below) pass because
  /// `raw` isn't valid/complete JSON on its own, not via a different code
  /// path.
  fn parse_members(raw: &str) -> Result<Option<BTreeMap<String, &RawValue>>, JsonParseError> {
    let value: &RawValue = serde_json::from_str(raw)?;
    if let Some(escape) = find_lone_surrogate(raw) {
      return Err(JsonParseError::Json(lone_surrogate_error(raw, escape)));
    }
    if !value.get().starts_with('{') {
      return Ok(None);
    }
    let duplicate: RefCell<Option<SmolStr>> = RefCell::new(None);
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    match deserializer.deserialize_map(MembersVisitor {
      duplicate: &duplicate,
    }) {
      Ok(members) => Ok(Some(members)),
      Err(err) => Err(match duplicate.into_inner() {
        Some(key) => JsonParseError::DuplicateField(key),
        None => JsonParseError::Json(err),
      }),
    }
  }

  /// [`Visitor`] behind [`parse_members`]'s second pass: the top-level
  /// object's members, each key decoded and each value captured as a
  /// [`RawValue`], with a member name seen earlier in the object refused.
  ///
  /// `MapAccess`'s error type is fixed to `serde_json::Error` by the
  /// driving `Deserializer`, which has no variant that names an arbitrary
  /// field, so the key travels out through `self.duplicate` instead —
  /// [`parse_members`] reads it back after the `Err` this returns
  /// propagates up through `deserialize_map`.
  struct MembersVisitor<'a> {
    duplicate: &'a RefCell<Option<SmolStr>>,
  }

  impl<'de> Visitor<'de> for MembersVisitor<'_> {
    type Value = BTreeMap<String, &'de RawValue>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("a JSON object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
      A: MapAccess<'de>,
    {
      let mut members = BTreeMap::new();
      while let Some(key) = map.next_key::<String>()? {
        if members.contains_key(&key) {
          *self.duplicate.borrow_mut() = Some(SmolStr::new(&key));
          return Err(de::Error::custom("duplicate top-level key"));
        }
        let value: &'de RawValue = map.next_value()?;
        members.insert(key, value);
      }
      Ok(members)
    }
  }

  /// The byte offset in `json` of the first `\u` escape that encodes one
  /// half of a UTF-16 surrogate pair without the other half, if any.
  ///
  /// `json` must be a text serde_json has accepted as JSON grammar. That
  /// grammar admits no backslash outside a string, and inside one every
  /// backslash opens an escape: two bytes (`\"`, `\\`, `\/`, `\b`, `\f`,
  /// `\n`, `\r`, `\t`) or `\u` and four hex digits. Stepping from escape to
  /// escape therefore meets every `\u` escape in every key and string at
  /// any depth, and parses nothing else. The rule is the one serde_json
  /// applies when it decodes a string into Rust text: a leading half
  /// (`\uD800` to `\uDBFF`) must be followed at once by a trailing half
  /// (`\uDC00` to `\uDFFF`), and a trailing half may appear nowhere else.
  fn find_lone_surrogate(json: &str) -> Option<usize> {
    let bytes = json.as_bytes();
    let mut index = 0;
    while let Some(found) = bytes.get(index..)?.iter().position(|&byte| byte == b'\\') {
      let escape = index + found;
      let Some(unit) = utf16_escape(bytes, escape) else {
        // A two-byte escape. Stepping over both bytes keeps the second
        // backslash of `\\` from passing as the start of an escape.
        index = escape + 2;
        continue;
      };
      index = escape + 6;
      match unit {
        0xD800..=0xDBFF => match utf16_escape(bytes, index) {
          Some(0xDC00..=0xDFFF) => index += 6,
          _ => return Some(escape),
        },
        0xDC00..=0xDFFF => return Some(escape),
        _ => {}
      }
    }
    None
  }

  /// The UTF-16 code unit of the `\uXXXX` escape at byte `at` of `bytes`,
  /// or `None` when no `\u` escape starts there.
  fn utf16_escape(bytes: &[u8], at: usize) -> Option<u16> {
    let digits = bytes.get(at..at.checked_add(6)?)?.strip_prefix(b"\\u")?;
    if !digits.iter().all(u8::is_ascii_hexdigit) {
      return None;
    }
    u16::from_str_radix(core::str::from_utf8(digits).ok()?, 16).ok()
  }

  /// The [`JsonParseError::Json`] error for the half surrogate pair whose
  /// escape starts at byte `escape` of `json`: serde_json's grammar admits
  /// the escape, but no Unicode text can hold it, so decoding the string
  /// it sits in fails. The error carries the escape's line and column,
  /// both counted from 1 as serde_json counts them.
  fn lone_surrogate_error(json: &str, escape: usize) -> serde_json::Error {
    let before = json.get(..escape).unwrap_or_default();
    let line = before.matches('\n').count() + 1;
    let column = before.len() - before.rfind('\n').map_or(0, |newline| newline + 1) + 1;
    de::Error::custom(format_args!(
      "unpaired UTF-16 surrogate in a string escape at line {line} column {column}"
    ))
  }

  /// A field's value, decoded in the JSON type [`Field::shape`] declares
  /// for it.
  enum FieldValue {
    /// [`Shape::String`]: the string as the answer wrote it.
    String(String),
    /// [`Shape::StringArray`]: the array's strings, in order, as the
    /// answer wrote them.
    Strings(Vec<String>),
  }

  impl FieldValue {
    /// A string field's value as [`ImageAnalysis`] holds it: the string,
    /// trimmed. `parse` asks this only of a string field's value, so the
    /// empty label for an array is never reached.
    fn into_label(self) -> SmolStr {
      match self {
        Self::String(string) => SmolStr::new(string.trim()),
        Self::Strings(_) => SmolStr::default(),
      }
    }

    /// An array field's value as [`ImageAnalysis`] holds it: each string
    /// through [`push_label`], whole, because a label can itself contain a
    /// comma (e.g. "red, white, and blue flag", "july 4, 2026"). `parse`
    /// asks this only of an array field's value, so the empty list for a
    /// string is never reached.
    fn into_labels(self) -> Vec<SmolStr> {
      let mut labels = Vec::new();
      if let Self::Strings(strings) = self {
        for string in &strings {
          push_label(&mut labels, string);
        }
      }
      labels
    }
  }

  /// `value` decoded in the JSON type `field`'s schema entry declares
  /// ([`Field::shape`]), or `None` when it holds any other: `null`, a
  /// boolean, a number, an object, a string where an array of strings is
  /// declared, or an array where a string is declared or that holds
  /// anything but strings. A JSON value's first byte names its type (`"`
  /// a string, `[` an array, `{` an object, `t` or `f` a boolean, `n`
  /// null, `-` or a digit a number), and only a value whose first byte
  /// names the declared type is decoded. So neither a number's magnitude
  /// nor a nested value's depth can fail a field: either is a wrong type,
  /// refused by name like any other.
  ///
  /// An `Err` would mean `value` is not JSON, which [`parse_members`] has
  /// already ruled out for the whole answer.
  fn decode_field(field: Field, value: &RawValue) -> Result<Option<FieldValue>, serde_json::Error> {
    Ok(match field.shape() {
      Shape::String => decode_string(value)?.map(FieldValue::String),
      Shape::StringArray => decode_strings(value)?.map(FieldValue::Strings),
    })
  }

  /// `value` decoded when it is a JSON string, else `None`.
  fn decode_string(value: &RawValue) -> Result<Option<String>, serde_json::Error> {
    let json = value.get();
    if json.starts_with('"') {
      serde_json::from_str(json).map(Some)
    } else {
      Ok(None)
    }
  }

  /// `value` decoded when it is a JSON array of strings, else `None`. Each
  /// element is captured as a [`RawValue`] and decoded only once its first
  /// byte shows a string.
  fn decode_strings(value: &RawValue) -> Result<Option<Vec<String>>, serde_json::Error> {
    let json = value.get();
    if !json.starts_with('[') {
      return Ok(None);
    }
    let items: Vec<&RawValue> = serde_json::from_str(json)?;
    let mut strings = Vec::with_capacity(items.len());
    for item in items {
      match decode_string(item)? {
        Some(string) => strings.push(string),
        None => return Ok(None),
      }
    }
    Ok(Some(strings))
  }

  /// Trims `raw`; pushes it onto `values` if non-empty and not already
  /// present (verbatim-case dedup — case-folding happens nowhere in
  /// this parser; see `parse_does_not_lowercase_labels` in the tests
  /// below).
  fn push_label(values: &mut Vec<SmolStr>, raw: &str) {
    let trimmed = raw.trim();
    if !trimmed.is_empty() && !values.iter().any(|existing| existing.as_str() == trimmed) {
      values.push(SmolStr::new(trimmed));
    }
  }

  /// `true` if `analysis` lacks the minimum content required to produce
  /// a useful indexing record. See [`ImageAnalysisTask::with_accept_empty`]
  /// for the full predicate this implements.
  fn lacks_indexable_content(analysis: &ImageAnalysis) -> bool {
    let has_prose_and_keywords = !analysis.description().is_empty() && !analysis.tags().is_empty();
    let has_substantive_detection = !analysis.subjects().is_empty()
      || !analysis.objects().is_empty()
      || !analysis.actions().is_empty();
    !has_prose_and_keywords && !has_substantive_detection
  }

  #[cfg(test)]
  mod tests {
    use super::*;
    use std::{format, vec};

    /// The task with every extension switched on: all ten fields. The
    /// regressions below that were written against the ten-field task run
    /// on it with their fixtures unchanged.
    fn full_task() -> ImageAnalysisTask {
      ImageAnalysisTask::new().with_extensions(Extension::ALL)
    }

    // ===== ported from both engine copies (lfm + qwen3-vl; the two
    // copies' test suites were substantively identical) =====

    /// The prompt must not enumerate value tokens — see
    /// `ImageAnalysisTask::build_prompt` for the presence-penalty
    /// rationale. The full roster's prompt holds every paragraph, so
    /// checking it checks them all.
    #[test]
    fn scene_prompt_does_not_enumerate_value_tokens() {
      let prompt_lower = full_task().prompt().to_lowercase();
      let banned_tokens = [
        "stage performance",
        "middle-aged man",
        "golden retriever",
        "birthday cake",
        "vintage red sports car",
        "cutting cake",
        "taking photos",
        "wide shot",
        "close-up",
        "medium shot",
        "over-the-shoulder",
        "celebratory",
        "natural light",
        "low light",
        "backlit",
      ];
      for token in banned_tokens {
        assert!(
          !prompt_lower.contains(&token.to_lowercase()),
          "the prompt must not enumerate value token {token:?} \
           (prompt-vocabulary tokens get a negative logit shift on \
           mistralrs-backed engines in deterministic mode); use \
           descriptive format guidance instead of `e.g. \"...\"` examples"
        );
      }
    }

    #[test]
    fn parse_valid_json() {
      let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"]}"#;
      let task = full_task();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.scene(), "beach");
      assert_eq!(result.description(), "Sunset over the ocean");
      assert_eq!(result.emotion().len(), 1);
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.categories(), &[SmolStr::from("nature")][..]);
    }

    #[test]
    fn reject_json_with_wrapper_text() {
      let text =
        "Here is the analysis:\n{\"scene\":\"office\",\"description\":\"People working\"}\nDone.";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    #[test]
    fn reject_plain_text_output() {
      let text = "A beautiful sunset over the ocean.";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    /// A fenced markdown block around an otherwise-valid JSON object is
    /// rejected the same way as prose-wrapped JSON: this parser expects
    /// `raw` to already be a bare JSON text, one object with nothing but
    /// JSON whitespace around it, and does not strip wrapping of any kind
    /// (fences included).
    #[test]
    fn reject_fenced_json() {
      let text = "```json\n{\"scene\":\"office\",\"description\":\"People working\"}\n```";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    /// LAW: the answer is a JSON text. The JSON whitespace a JSON text may
    /// carry around its value parses; any other whitespace there (a
    /// no-break space, a line separator, an ideographic space, a vertical
    /// tab) is not JSON and is refused as `JsonParseError::Json`, as a
    /// fence or wrapper prose is.
    #[test]
    fn only_json_whitespace_may_surround_the_answer() {
      let task = ImageAnalysisTask::new();
      let object = format!("{{{DEFAULT_MEMBERS}}}");
      task
        .parse(&format!(" \t\r\n{object} \t\r\n"))
        .expect("JSON whitespace around the object is part of a JSON text");
      for framed in [
        format!("\u{a0}{object}"),
        format!("{object}\u{a0}"),
        format!("{object}\u{2028}"),
        format!("\u{3000}{object}"),
        format!("\u{b}{object}"),
      ] {
        assert!(
          matches!(task.parse(&framed), Err(JsonParseError::Json(_))),
          "{framed:?} is not a JSON text"
        );
      }
    }

    /// LAW: `tags` given as one comma-separated string is refused by name.
    /// The schema declares an array of strings, and `parse` never splits a
    /// string into labels.
    #[test]
    fn reject_comma_separated_tag_string() {
      let json = r#"{"description":"A singer on stage","tags":"concert, live music, spotlight"}"#;
      let task = ImageAnalysisTask::new();
      match task.parse(json) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["tags"]),
        other => panic!("expected MissingFields naming tags alone, got {other:?}"),
      }
    }

    #[test]
    fn reject_empty_json_payload() {
      let task = ImageAnalysisTask::new();
      assert!(task.parse("{}").is_err());
    }

    /// Repairs a vacuous fixture (Codex R1, PR #5): the original object
    /// omitted eight of the nine required fields, so `.is_err()` passed
    /// for the wrong reason — `MissingFields` naming the absent required
    /// fields, never reaching the unknown-key check at all. This fixture
    /// is otherwise-complete (all ten declared properties present and
    /// well-shaped) plus exactly one undeclared key, so the ONLY
    /// violation possible is the unknown key, and the assertion now
    /// pins the specific error variant instead of any error.
    #[test]
    fn reject_unknown_json_fields() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":["business"],"extra":"unexpected"}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("a key outside the ten declared properties must be rejected");
      match err {
        JsonParseError::UnknownFields(fields) => assert!(
          fields.iter().any(|f| f == "extra"),
          "expected 'extra' named in {fields:?}"
        ),
        other => panic!("expected UnknownFields naming extra, got {other:?}"),
      }
    }

    // ===== duplicate top-level key regressions (Codex R2, PR #5) =====
    //
    // Every fixture below is otherwise-complete (all ten declared
    // properties present and well-shaped, same discipline as
    // `reject_unknown_json_fields` above) plus exactly one duplicated
    // top-level key, so the ONLY violation possible is the duplicate
    // itself, and each assertion pins `JsonParseError::DuplicateField`
    // rather than any error.

    /// `categories: null` then `categories: [...]` — before this fix,
    /// last-write-wins collapse silently kept the *valid* second copy,
    /// so this exact order parsed successfully with no error at all
    /// (the bug Codex R2 named: `{"categories": null, "categories":
    /// []}` used to pass).
    #[test]
    fn reject_duplicate_categories_null_then_valid() {
      let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":null,"categories":["nature"]}"#;
      let task = full_task();
      let err = task.parse(json).expect_err(
        "a duplicated top-level key must be rejected regardless of which copy survives collapse",
      );
      match err {
        JsonParseError::DuplicateField(key) => assert_eq!(key, "categories"),
        other => panic!("expected DuplicateField naming categories, got {other:?}"),
      }
    }

    /// The reverse order: `categories: [...]` then `categories: null`.
    /// Before this fix, collapse kept the *null* second copy, so the
    /// field check already rejected this order (as
    /// `MissingFields`) — the asymmetry Codex R2 flagged. Both orders
    /// must now be refused the same way, for the same reason, this
    /// early: `DuplicateField`, not `MissingFields`.
    #[test]
    fn reject_duplicate_categories_valid_then_null() {
      let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"],"categories":null}"#;
      let task = full_task();
      let err = task.parse(json).expect_err(
        "a duplicated top-level key must be rejected regardless of which copy survives collapse",
      );
      match err {
        JsonParseError::DuplicateField(key) => assert_eq!(key, "categories"),
        other => panic!("expected DuplicateField naming categories, got {other:?}"),
      }
    }

    /// A wrong-typed duplicate of a required field, wrong-type first:
    /// `scene: 42` then `scene: "beach"`. Before this fix, collapse
    /// kept the *valid* second copy, so this order parsed successfully
    /// with no error — the same silent-bypass shape as the null case
    /// above, but for a type violation instead of a missing value.
    #[test]
    fn reject_duplicate_required_field_wrong_type_then_valid() {
      let json = r#"{"scene":42,"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"]}"#;
      let task = full_task();
      let err = task.parse(json).expect_err(
        "a duplicated top-level key must be rejected regardless of which copy survives collapse",
      );
      match err {
        JsonParseError::DuplicateField(key) => assert_eq!(key, "scene"),
        other => panic!("expected DuplicateField naming scene, got {other:?}"),
      }
    }

    /// The reverse order: `scene: "beach"` then `scene: 42`. Before
    /// this fix, collapse kept the *wrong-typed* second copy, so the
    /// field check already rejected this order (as
    /// `MissingFields`). Both orders must now be refused the same way,
    /// this early: `DuplicateField`, not `MissingFields`.
    #[test]
    fn reject_duplicate_required_field_valid_then_wrong_type() {
      let json = r#"{"scene":"beach","scene":42,"description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"]}"#;
      let task = full_task();
      let err = task.parse(json).expect_err(
        "a duplicated top-level key must be rejected regardless of which copy survives collapse",
      );
      match err {
        JsonParseError::DuplicateField(key) => assert_eq!(key, "scene"),
        other => panic!("expected DuplicateField naming scene, got {other:?}"),
      }
    }

    /// A duplicated key OUTSIDE the ten declared properties: before
    /// this fix, collapse would still have left exactly one `extra`
    /// key for `unknown_fields` to name (duplicating an unknown key
    /// doesn't change ITS value being unusable), so this specific
    /// shape wasn't a silent-pass bug — but it pins that the duplicate
    /// check runs on every top-level key, declared or not, and fires
    /// as `DuplicateField` before `unknown_fields` ever sees the
    /// (already-collapsed-to-one) key.
    #[test]
    fn reject_duplicate_unknown_key() {
      let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"],"extra":"foo","extra":"bar"}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("a duplicated top-level key must be rejected even when the name is outside the declared schema");
      match err {
        JsonParseError::DuplicateField(key) => assert_eq!(key, "extra"),
        other => panic!("expected DuplicateField naming extra, got {other:?}"),
      }
    }

    #[test]
    fn reject_missing_required_fields() {
      let json = r#"{"description":"A singer on stage","tags":["concert"]}"#;
      let task = full_task();
      assert!(task.parse(json).is_err());
    }

    #[test]
    fn parse_array_form_subjects() {
      let json_list = r#"{"description":"y","subjects":["a","b"],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new().with_extensions([Extension::Subjects]);
      let result = task.parse(json_list).expect("list-form parse");
      assert_eq!(result.subjects().len(), 2);
      assert_eq!(result.subjects()[0], "a");
      assert_eq!(result.subjects()[1], "b");
    }

    /// LAW: `subjects` given as a bare string is refused by name. The
    /// schema declares an array of strings, and `parse` never wraps a
    /// string into one.
    #[test]
    fn reject_subjects_string_form() {
      let json = r#"{"scene":"x","description":"y","subjects":"middle-aged man, in red jacket","objects":[],"actions":[],"emotion":[],"shot_type":"x","lighting":[],"tags":["t"],"categories":[]}"#;
      let task = full_task();
      match task.parse(json) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["subjects"]),
        other => panic!("expected MissingFields naming subjects alone, got {other:?}"),
      }
    }

    #[test]
    fn reject_all_required_fields_empty_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject all-empty payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn accept_all_required_fields_empty_payload_when_opted_in() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task().with_accept_empty(true);
      let result = task
        .parse(json)
        .expect("opt-in must accept the all-empty payload");
      assert!(result.scene().is_empty());
      assert!(result.description().is_empty());
      assert!(result.subjects().is_empty());
      assert!(result.objects().is_empty());
      assert!(result.actions().is_empty());
      assert!(result.emotion().is_empty());
      assert!(result.shot_type().is_empty());
      assert!(result.lighting().is_empty());
      assert!(result.tags().is_empty());
      assert!(result.categories().is_empty());
    }

    #[test]
    fn reject_tags_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": ["concert", "live music"],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject tags-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_scene_only_payload_by_default() {
      let json = r#"{
            "scene": "office",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject scene-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_description_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "People working in an office",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject description-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn accept_minimal_indexable_payload() {
      let json = r#"{
            "scene": "",
            "description": "Two people talking",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": ["conversation"],
            "categories": []
          }"#;
      let task = full_task();
      let result = task
        .parse(json)
        .expect("description+tags must clear the indexable threshold");
      assert_eq!(result.description(), "Two people talking");
      assert_eq!(result.tags(), &[SmolStr::from("conversation")][..]);
      assert!(result.subjects().is_empty());
      assert!(result.objects().is_empty());
      assert!(result.scene().is_empty());
    }

    #[test]
    fn accept_detection_rich_payload_with_empty_description_and_tags() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": ["middle-aged woman in red dress"],
            "objects": ["wedding cake"],
            "actions": ["cutting cake"],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let result = task.parse(json).expect(
        "detection-rich payload must clear the indexable threshold via \
               the detection-bucket path even when description+tags are empty",
      );
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.objects().len(), 1);
      assert_eq!(result.actions().len(), 1);
      assert!(result.description().is_empty());
      assert!(result.tags().is_empty());
    }

    #[test]
    fn accept_subjects_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": ["a single subject label"],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let result = task
        .parse(json)
        .expect("subjects-only must clear the indexable threshold");
      assert_eq!(result.subjects().len(), 1);
    }

    #[test]
    fn accept_objects_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": ["a single object label"],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let result = task
        .parse(json)
        .expect("objects-only must clear the indexable threshold");
      assert_eq!(result.objects().len(), 1);
    }

    #[test]
    fn accept_actions_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": ["a single action label"],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let result = task
        .parse(json)
        .expect("actions-only must clear the indexable threshold");
      assert_eq!(result.actions().len(), 1);
    }

    #[test]
    fn reject_emotion_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": ["calm"],
            "shot_type": "",
            "lighting": [],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject emotion-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_lighting_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": ["natural light"],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject lighting-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_attribute_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": ["tense"],
            "shot_type": "",
            "lighting": ["low light"],
            "tags": [],
            "categories": []
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("style-attribute-only payload must reject regardless of bucket count");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_null_required_array() {
      let json = r#"{
            "scene": "office",
            "description": "people working",
            "subjects": null,
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("null required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(
            fields.contains(&"subjects"),
            "expected 'subjects' in MissingFields, got {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn reject_null_required_string() {
      let json = r#"{
            "scene": null,
            "description": "people working",
            "subjects": ["person"],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("null required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(
            fields.contains(&"scene"),
            "expected 'scene' in MissingFields, got {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn reject_multiple_null_required_fields() {
      let json = r#"{
            "scene": null,
            "description": null,
            "subjects": null,
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("null required fields must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(fields.contains(&"scene"), "missing 'scene' in {fields:?}");
          assert!(
            fields.contains(&"description"),
            "missing 'description' in {fields:?}"
          );
          assert!(
            fields.contains(&"subjects"),
            "missing 'subjects' in {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn array_elements_are_not_comma_split() {
      let json = r#"{
            "scene": "patriotic event",
            "description": "Flag display",
            "subjects": ["middle-aged man, in red jacket"],
            "objects": ["red, white, and blue flag", "birthday cake with candles, balloons"],
            "actions": ["waving"],
            "emotion": ["festive"],
            "shot_type": "wide shot",
            "lighting": ["natural, dramatic backlight"],
            "tags": ["july 4, 2026"],
            "categories": ["celebration"]
          }"#;
      let task = full_task();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
      assert_eq!(result.objects().len(), 2);
      assert_eq!(result.objects()[0], "red, white, and blue flag");
      assert_eq!(result.objects()[1], "birthday cake with candles, balloons");
      assert_eq!(result.lighting().len(), 1);
      assert_eq!(result.lighting()[0], "natural, dramatic backlight");
      assert_eq!(result.tags().len(), 1);
      assert_eq!(result.tags()[0].as_str(), "july 4, 2026");
    }

    /// LAW: `shot_type` given as a one-element array is refused by name.
    /// The schema declares a string, and `parse` never unwraps an array.
    #[test]
    fn reject_shot_type_list_form() {
      let json_one = r#"{"description":"y","shot_type":["wide shot"],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new().with_extensions([Extension::ShotType]);
      match task.parse(json_one) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["shot_type"]),
        other => panic!("expected MissingFields naming shot_type alone, got {other:?}"),
      }
    }

    /// A multi-element `shot_type` array is now a named
    /// `MissingFields(["shot_type"])` error rather than the generic
    /// serde message the two engine copies produced ("expected a
    /// single shot_type label, got multiple values", wrapped as
    /// `JsonParseError::Json`) — folded into the field check
    /// (`usable_value`) like every other shape violation.
    #[test]
    fn reject_shot_type_multi_element_array_with_named_error() {
      let json_many = r#"{"scene":"x","description":"y","subjects":[],"objects":[],"actions":[],"emotion":[],"shot_type":["wide","close-up"],"lighting":[],"tags":["t"]}"#;
      let task = full_task();
      let err = task
        .parse(json_many)
        .expect_err("multi-element shot_type array must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(fields.contains(&"shot_type")),
        other => panic!("expected MissingFields naming shot_type, got {other:?}"),
      }
    }

    // ===== new for the llmtask 0.3 merge: categories, the sealed label
    // discipline, and the wrong-type / non-object resilience cases the
    // brief calls for =====

    /// `categories` was the one field this Task marked optional through
    /// 0.3.0's initial merge; this test used to be
    /// `categories_absent_defaults_to_empty`, pinning that an object
    /// missing `categories` still parsed. The owner overturned the
    /// optional ruling in the same round that addressed Codex R1
    /// (PR #5): `categories` is now required like the other nine, so
    /// this regression is flipped — absence is now refused, the same as
    /// omitting any other required field. Under the field roster,
    /// `categories` is an extension; once switched on, it is required like
    /// every field the task asks for.
    #[test]
    fn categories_absent_is_rejected() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"]}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("categories is now required; absence must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"categories"),
          "expected 'categories' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming categories, got {other:?}"),
      }
    }

    /// Codex R1 (PR #5): the field check used to exempt a *present*
    /// `categories: null` from the shape check (it fell through a guard
    /// that only made sense back when a bare `!matches!(v, Value::Null)`
    /// meant "is this key even here" — categories was still optional at
    /// the time), silently defaulting it to an empty list even though
    /// the schema's `categories` entry allows only an array of strings,
    /// never null. Now that `categories` is a required field (see
    /// `categories_absent_is_rejected` above), a present `null` and a
    /// totally absent key produce the same named error through two
    /// branches of `usable_value` (an absent member, and a value whose
    /// first byte is not the declared type's) — this test keeps the
    /// present-null input shape pinned separately from the absent-key
    /// shape so a future regression in either branch is still caught.
    #[test]
    fn reject_present_null_categories_with_named_error() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":null}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("a present null categories must be rejected, not silently defaulted");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"categories"),
          "expected 'categories' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming categories, got {other:?}"),
      }
    }

    #[test]
    fn categories_present_populates() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":["business","corporate"]}"#;
      let task = full_task();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(
        result.categories(),
        &[SmolStr::from("business"), SmolStr::from("corporate")][..]
      );
    }

    #[test]
    fn reject_wrong_type_required_field_with_named_error() {
      let json = r#"{"scene":"office","description":"people working","subjects":42,"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"]}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("wrong-type required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"subjects"),
          "expected 'subjects' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming subjects, got {other:?}"),
      }
    }

    // Renamed from `reject_wrong_type_optional_field_with_named_error`:
    // `categories` is no longer this Task's optional field (see
    // `categories_absent_is_rejected` above), so the old name's
    // "optional" no longer describes it — the fixture and assertion are
    // unchanged, a wrong-type `categories` was already rejected the same
    // way before the required/optional ruling changed.
    #[test]
    fn reject_wrong_type_categories_field_with_named_error() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":42}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("wrong-type categories field must still be rejected, not silently dropped");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"categories"),
          "expected 'categories' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming categories, got {other:?}"),
      }
    }

    /// Class-sweep addition (R1 follow-through, not itself a Codex
    /// finding): `decode_field` decodes an array field only when every
    /// element is a string, enforcing the schema's
    /// `items: {"type": "string"}` promise for every array-shaped field.
    /// That was already implemented but had no element-level regression
    /// — only a whole-field wrong type (`subjects: 42`, above) had
    /// coverage, which exercises a different branch of `decode_field`
    /// than a well-typed array with one bad element does.
    #[test]
    fn reject_array_field_with_non_string_element() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person",42],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"]}"#;
      let task = full_task();
      let err = task
        .parse(json)
        .expect_err("an array field with a non-string element must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"subjects"),
          "expected 'subjects' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming subjects, got {other:?}"),
      }
    }

    /// A non-object answer lacks every field the task asks for, and names
    /// exactly those: the default task's two, the full roster's ten.
    #[test]
    fn reject_non_object_top_level_value() {
      for (task, asked_for) in [
        (ImageAnalysisTask::new(), vec!["description", "tags"]),
        (full_task(), Field::ALL.map(Field::key).to_vec()),
      ] {
        let err = task
          .parse(r#"["not", "an", "object"]"#)
          .expect_err("a JSON array at the top level must be rejected");
        match err {
          JsonParseError::MissingFields(fields) => assert_eq!(fields, asked_for),
          other => panic!("expected MissingFields naming every asked-for field, got {other:?}"),
        }
      }
    }

    /// Sealed discipline (see `PROMPT_RULES`, which closes every
    /// roster's prompt): every array field's elements must be lowercase,
    /// singular, 1-3-word phrases with no trailing punctuation — as a
    /// PROMPT instruction. This test pins the prompt text; the parser
    /// itself must never enforce it by transforming output (see
    /// `parse_does_not_lowercase_labels` below) — that's the model's
    /// job, not the parser's.
    #[test]
    fn label_array_discipline_is_stated_in_prompt() {
      for task in [ImageAnalysisTask::new(), full_task()] {
        assert!(
          task
            .prompt()
            .contains("lowercase, singular, 1-3-word phrases"),
          "prompt must instruct the sealed per-array label discipline verbatim"
        );
        assert!(
          task.prompt().contains("no trailing punctuation"),
          "prompt must instruct no trailing punctuation on array elements"
        );
      }
    }

    /// Verbatim law: the parser must NEVER lowercase (or otherwise
    /// case-fold) label text at parse time, even though the prompt asks
    /// the model for lowercase output. The discipline is prompt-only;
    /// enforcing it in the parser would silently mask a
    /// non-compliant model instead of surfacing the drift.
    #[test]
    fn parse_does_not_lowercase_labels() {
      let json = r#"{"scene":"Office","description":"Desc","subjects":["MidCase Person"],"shot_type":"Wide Shot","tags":["MixedCase"]}"#;
      let task = ImageAnalysisTask::new().with_extensions([
        Extension::Scene,
        Extension::Subjects,
        Extension::ShotType,
      ]);
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.scene(), "Office");
      assert_eq!(result.shot_type(), "Wide Shot");
      assert_eq!(result.subjects()[0], "MidCase Person");
      assert_eq!(result.tags()[0], "MixedCase");
    }

    /// The grammar/regex face constrains array items to plain strings
    /// only — no `pattern` regex forcing lowercase/word-count at the
    /// schema level. The discipline lives in the prompt (pinned above),
    /// never in the constrained-decoding grammar.
    #[test]
    fn array_schema_items_are_plain_strings_only() {
      let task = full_task();
      let schema = task.schema();
      for field in [
        "subjects",
        "objects",
        "actions",
        "emotion",
        "lighting",
        "tags",
        "categories",
      ] {
        let items = &schema["properties"][field]["items"];
        assert_eq!(
          items["type"], "string",
          "field {field} items must be plain strings"
        );
        assert!(
          items.get("pattern").is_none(),
          "field {field} items must not carry a regex pattern constraint"
        );
      }
    }

    // ===== Task contract round-trip =====

    #[test]
    fn prompt_is_non_empty() {
      assert!(!ImageAnalysisTask::new().prompt().is_empty());
    }

    #[test]
    fn schema_is_valid_json_schema_shape() {
      let task = full_task();
      let schema = task.schema();
      assert_eq!(schema["type"], "object");
      assert_eq!(schema["additionalProperties"], false);
      let properties = schema["properties"]
        .as_object()
        .expect("properties must be an object");
      assert_eq!(
        properties.len(),
        10,
        "every extension on: all ten ImageAnalysis fields"
      );
      let required = schema["required"]
        .as_array()
        .expect("required must be an array");
      assert_eq!(required.len(), Field::ALL.len());
      assert_eq!(required.len(), 10, "all ten fields are required");
      assert!(
        required.iter().any(|v| v == "categories"),
        "categories must be required, per the owner's ruling overturning the earlier optional choice"
      );
    }

    #[test]
    fn grammar_wraps_the_cached_schema() {
      let task = ImageAnalysisTask::new();
      let grammar = task.grammar();
      assert!(grammar.is_json_schema());
      assert_eq!(grammar.as_json_schema(), Some(task.schema()));
    }

    // ===== the field roster and the caps =====

    /// The paragraph each extension adds to the prompt: the ten-field
    /// prompt's own wording for that field, unchanged.
    const EXTENSION_PARAGRAPHS: [(Extension, &str); 8] = [
      (
        Extension::Scene,
        "scene: a single short scene-category label in lowercase English, 1-3 words, no full sentence.",
      ),
      (
        Extension::Subjects,
        "subjects: array of distinct people or animals with visible distinguishing features.",
      ),
      (
        Extension::Objects,
        "objects: array of notable, search-relevant objects.",
      ),
      (Extension::Actions, "actions: array of visible actions."),
      (
        Extension::Emotion,
        "emotion: array of descriptors for the scene's overall emotional tone.",
      ),
      (
        Extension::ShotType,
        "shot_type: a single short camera-shot label in lowercase English, 1-2 words (a cinematography term).",
      ),
      (
        Extension::Lighting,
        "lighting: array of lighting descriptors.",
      ),
      (
        Extension::Categories,
        "categories: array of broad content categories, coarser-grained than tags.",
      ),
    ];

    /// The two fields every task asks for, as a well-shaped answer
    /// spells them (without the enclosing braces).
    const DEFAULT_MEMBERS: &str =
      r#""description":"A person reads by a window.","tags":["reading"]"#;

    fn nz(n: usize) -> NonZeroUsize {
      NonZeroUsize::new(n).expect("a cap is never zero")
    }

    /// A well-shaped JSON value for `extension`'s field that parses to the
    /// single label `sample`.
    fn sample(extension: Extension) -> &'static str {
      match extension {
        Extension::Scene | Extension::ShotType => r#""sample""#,
        _ => r#"["sample"]"#,
      }
    }

    /// The labels the parsed analysis holds in `extension`'s field; empty
    /// when the field is empty.
    fn read(analysis: &ImageAnalysis, extension: Extension) -> Vec<&str> {
      let labels: Vec<&str> = match extension {
        Extension::Scene => vec![analysis.scene()],
        Extension::ShotType => vec![analysis.shot_type()],
        Extension::Subjects => analysis.subjects().iter().map(SmolStr::as_str).collect(),
        Extension::Objects => analysis.objects().iter().map(SmolStr::as_str).collect(),
        Extension::Actions => analysis.actions().iter().map(SmolStr::as_str).collect(),
        Extension::Emotion => analysis.emotion().iter().map(SmolStr::as_str).collect(),
        Extension::Lighting => analysis.lighting().iter().map(SmolStr::as_str).collect(),
        Extension::Categories => analysis.categories().iter().map(SmolStr::as_str).collect(),
      };
      labels
        .into_iter()
        .filter(|label| !label.is_empty())
        .collect()
    }

    /// The extensions are keyed by the contract's own field names, in
    /// `ImageAnalysis` field order.
    #[test]
    fn extension_keys_are_the_contract_field_names() {
      assert_eq!(
        Extension::ALL.map(|extension| extension.as_str()),
        [
          "scene",
          "subjects",
          "objects",
          "actions",
          "emotion",
          "shot_type",
          "lighting",
          "categories"
        ]
      );
    }

    /// LAW: the default task's schema names exactly `description` and
    /// `tags`, and requires both.
    #[test]
    fn default_schema_names_exactly_description_and_tags() {
      let task = ImageAnalysisTask::new();
      let schema = task.schema();
      let mut properties: Vec<&str> = schema["properties"]
        .as_object()
        .expect("properties must be an object")
        .keys()
        .map(String::as_str)
        .collect();
      properties.sort_unstable();
      assert_eq!(properties, ["description", "tags"]);
      assert_eq!(schema["required"], json!(["description", "tags"]));
      assert_eq!(schema["type"], "object");
      assert_eq!(schema["additionalProperties"], false);
      for extension in Extension::ALL {
        assert!(
          !task.has_extension(extension),
          "{extension:?} must be off by default"
        );
      }
    }

    /// LAW: an extension switches its field on in the schema, the prompt
    /// and `parse` together. Off, the field is in none of the three, and an
    /// answer carrying it is refused as undeclared; on, the schema declares
    /// and requires it, the prompt describes it in its unchanged wording,
    /// and `parse` reads it.
    #[test]
    fn an_extension_switches_its_field_on_in_schema_prompt_and_parse_together() {
      for (extension, paragraph) in EXTENSION_PARAGRAPHS {
        let key = extension.as_str();
        let answer = format!(r#"{{{DEFAULT_MEMBERS},"{key}":{}}}"#, sample(extension));

        let off = ImageAnalysisTask::new();
        let required = off.schema()["required"]
          .as_array()
          .expect("required must be an array");
        assert!(
          off.schema()["properties"].get(key).is_none(),
          "{key} off: the schema must not declare it"
        );
        assert!(
          !required.iter().any(|name| name == key),
          "{key} off: the schema must not require it"
        );
        assert!(
          !off.prompt().contains(&format!("\n{key}:")),
          "{key} off: the prompt must not describe it"
        );
        match off.parse(&answer) {
          Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, [key]),
          other => panic!("{key} off: expected UnknownFields naming it, got {other:?}"),
        }

        let on = ImageAnalysisTask::new().with_extensions([extension]);
        let required = on.schema()["required"]
          .as_array()
          .expect("required must be an array");
        assert!(on.has_extension(extension));
        assert!(
          on.schema()["properties"].get(key).is_some(),
          "{key} on: the schema must declare it"
        );
        assert!(
          required.iter().any(|name| name == key),
          "{key} on: the schema must require it"
        );
        assert!(
          on.prompt().contains(&format!("\n{paragraph}\n")),
          "{key} on: the prompt must describe it in its unchanged wording"
        );
        let analysis = on
          .parse(&answer)
          .unwrap_or_else(|err| panic!("{key} on: the answer must parse, got {err:?}"));
        assert_eq!(read(&analysis, extension), ["sample"]);
      }
    }

    /// LAW: an answer that leaves out every field the task did not ask for
    /// parses, and each of those fields reads empty.
    #[test]
    fn a_disabled_field_absent_from_the_answer_parses() {
      let analysis = ImageAnalysisTask::new()
        .parse(&format!("{{{DEFAULT_MEMBERS}}}"))
        .expect("the default task asks for description and tags only");
      assert_eq!(analysis.description(), "A person reads by a window.");
      assert_eq!(analysis.tags(), &[SmolStr::from("reading")][..]);
      for extension in Extension::ALL {
        assert!(
          read(&analysis, extension).is_empty(),
          "{extension:?} was not asked for and must read empty"
        );
      }
    }

    /// LAW: a field the task asks for, absent from the answer, is
    /// `MissingFields` naming that field and no other.
    #[test]
    fn an_enabled_field_absent_is_missing_fields_naming_it_alone() {
      for extension in Extension::ALL {
        let task = ImageAnalysisTask::new().with_extensions([extension]);
        match task.parse(&format!("{{{DEFAULT_MEMBERS}}}")) {
          Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, [extension.as_str()]),
          other => {
            panic!("{extension:?} on: expected MissingFields naming it alone, got {other:?}")
          }
        }
      }
      let task = ImageAnalysisTask::new();
      for (answer, missing) in [
        (r#"{"tags":["reading"]}"#, "description"),
        (r#"{"description":"A person reads by a window."}"#, "tags"),
      ] {
        match task.parse(answer) {
          Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, [missing]),
          other => panic!("expected MissingFields naming {missing} alone, got {other:?}"),
        }
      }
    }

    /// LAW: the caps appear in the schema, `maxLength` on `description` and
    /// `maxItems` on `tags`, at the named defaults unless a builder sets
    /// them. No other field carries a cap.
    #[test]
    fn the_caps_appear_in_the_schema() {
      let task = ImageAnalysisTask::new();
      assert_eq!(
        task.description_max_chars(),
        ImageAnalysisTask::DEFAULT_DESCRIPTION_MAX_CHARS
      );
      assert_eq!(
        task.tags_max_items(),
        ImageAnalysisTask::DEFAULT_TAGS_MAX_ITEMS
      );
      let properties = &task.schema()["properties"];
      assert_eq!(
        properties["description"]["maxLength"],
        ImageAnalysisTask::DEFAULT_DESCRIPTION_MAX_CHARS.get()
      );
      assert_eq!(
        properties["tags"]["maxItems"],
        ImageAnalysisTask::DEFAULT_TAGS_MAX_ITEMS.get()
      );

      let task = full_task()
        .with_description_max_chars(nz(40))
        .with_tags_max_items(nz(3));
      let properties = task.schema()["properties"]
        .as_object()
        .expect("properties must be an object");
      assert_eq!(properties["description"]["maxLength"], 40);
      assert_eq!(properties["tags"]["maxItems"], 3);
      for (key, property) in properties {
        if key != "description" {
          assert!(
            property.get("maxLength").is_none(),
            "{key} carries no maxLength"
          );
        }
        if key != "tags" {
          assert!(
            property.get("maxItems").is_none(),
            "{key} carries no maxItems"
          );
        }
      }
    }

    /// The caps appear in the prompt too: the description is one sentence
    /// of at most the cap's characters, and `tags` lists at most the cap's
    /// labels.
    #[test]
    fn the_caps_appear_in_the_prompt() {
      let task = ImageAnalysisTask::new();
      assert!(task.prompt().contains(&format!(
        "\ndescription: one concise sentence in English, at most {} characters, describing",
        ImageAnalysisTask::DEFAULT_DESCRIPTION_MAX_CHARS
      )));
      assert!(task.prompt().contains(&format!(
        "\ntags: array of at most {} short English search tags.",
        ImageAnalysisTask::DEFAULT_TAGS_MAX_ITEMS
      )));

      let task = task
        .with_description_max_chars(nz(40))
        .with_tags_max_items(nz(3));
      assert!(task.prompt().contains("at most 40 characters"));
      assert!(
        task
          .prompt()
          .contains("at most 3 short English search tags")
      );
    }

    /// An answer over a cap is refused by name, as a missing field is,
    /// and an answer at a cap parses whole: `parse` never truncates.
    /// `description` is counted in characters, not bytes, and as the answer
    /// wrote it; `tags` in the elements of its array.
    #[test]
    fn an_answer_over_a_cap_is_refused_by_name_and_never_truncated() {
      let task = ImageAnalysisTask::new()
        .with_description_max_chars(nz(10))
        .with_tags_max_items(nz(3));
      let refusal = |answer: &str| match task.parse(answer) {
        Err(JsonParseError::MissingFields(fields)) => fields,
        other => panic!("expected MissingFields for {answer}, got {other:?}"),
      };

      let at_cap = "é".repeat(10);
      let analysis = task
        .parse(&format!(r#"{{"description":"{at_cap}","tags":["a"]}}"#))
        .expect("a description of ten two-byte characters is at the cap");
      assert_eq!(analysis.description(), at_cap);
      let over_cap = "é".repeat(11);
      assert_eq!(
        refusal(&format!(r#"{{"description":"{over_cap}","tags":["a"]}}"#)),
        ["description"]
      );
      assert_eq!(
        refusal(&format!(r#"{{"description":"{at_cap} ","tags":["a"]}}"#)),
        ["description"],
        "a trailing space the schema counts is over the cap before parse trims it"
      );

      let analysis = task
        .parse(r#"{"description":"a","tags":["x","y","z"]}"#)
        .expect("three tags are at the cap");
      assert_eq!(analysis.tags().len(), 3);
      assert_eq!(
        refusal(r#"{"description":"a","tags":["x","y","z","w"]}"#),
        ["tags"]
      );
    }

    /// `accept_empty` keeps its meaning on the default task: an answer that
    /// does not populate both `description` and `tags` is a regression
    /// unless the task opts in.
    #[test]
    fn accept_empty_keeps_its_meaning_on_the_default_task() {
      for answer in [
        r#"{"description":"","tags":[]}"#,
        r#"{"description":"A person reads by a window.","tags":[]}"#,
        r#"{"description":"","tags":["reading"]}"#,
      ] {
        let task = ImageAnalysisTask::new();
        assert!(
          matches!(task.parse(answer), Err(JsonParseError::NoUsableFields)),
          "{answer} must be NoUsableFields by default"
        );
        assert!(
          task.with_accept_empty(true).parse(answer).is_ok(),
          "{answer} must parse when accept_empty is on"
        );
      }
    }

    // ===== every field in exactly the shape its schema entry declares =====

    /// A full-roster answer: every field present, non-empty, and of the
    /// JSON type its schema entry declares.
    const FULL_ANSWER: [(&str, &str); 10] = [
      ("scene", r#""office""#),
      ("description", r#""People work at their desks.""#),
      ("subjects", r#"["office worker"]"#),
      ("objects", r#"["desk"]"#),
      ("actions", r#"["typing"]"#),
      ("emotion", r#"["calm"]"#),
      ("shot_type", r#""wide""#),
      ("lighting", r#"["daylight"]"#),
      ("tags", r#"["office"]"#),
      ("categories", r#"["work"]"#),
    ];

    /// The keys of [`FULL_ANSWER`] whose type the full roster's emitted
    /// schema declares as `declared` (`"string"` or `"array"`), in field
    /// order.
    fn keys_declared_as(declared: &str) -> Vec<&'static str> {
      let task = full_task();
      FULL_ANSWER
        .iter()
        .map(|&(key, _)| key)
        .filter(|key| task.schema()["properties"][*key]["type"].as_str() == Some(declared))
        .collect()
    }

    /// [`FULL_ANSWER`] with `key`'s value replaced by `value`.
    fn full_answer_with(key: &str, value: &str) -> String {
      let members = FULL_ANSWER.map(|(name, canonical)| {
        format!(
          r#""{name}":{}"#,
          if name == key { value } else { canonical }
        )
      });
      format!("{{{}}}", members.join(","))
    }

    /// Asserts that the full roster refuses [`FULL_ANSWER`] with `key`'s
    /// value replaced by `value` as `MissingFields` naming `key` alone.
    fn assert_refused_by_name(key: &str, value: &str) {
      match full_task().parse(&full_answer_with(key, value)) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(
          fields,
          [key],
          "{key} = {value} must be refused naming {key} alone"
        ),
        other => panic!("{key} = {value} must be MissingFields naming {key} alone, got {other:?}"),
      }
    }

    /// The baseline every refusal law below changes one field of: each value
    /// of [`FULL_ANSWER`] has the JSON type the emitted schema declares for
    /// its key, and the whole answer parses.
    #[test]
    fn the_full_answer_parses_in_every_declared_shape() {
      let task = full_task();
      for (key, value) in FULL_ANSWER {
        let value: Value = serde_json::from_str(value).expect("a JSON value");
        let declared = task.schema()["properties"][key]["type"].as_str();
        let of_declared_type = match declared {
          Some("string") => value.is_string(),
          Some("array") => value
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_string)),
          _ => false,
        };
        assert!(
          of_declared_type,
          "{key}: {value} must be of the declared type {declared:?}"
        );
      }
      let members = FULL_ANSWER.map(|(key, value)| format!(r#""{key}":{value}"#));
      let analysis = task
        .parse(&format!("{{{}}}", members.join(",")))
        .expect("every field in its declared shape parses");
      assert_eq!(analysis.shot_type(), "wide");
      assert_eq!(analysis.subjects(), &[SmolStr::from("office worker")][..]);
      assert_eq!(analysis.tags(), &[SmolStr::from("office")][..]);
    }

    /// LAW: a field the schema declares as an array of strings is refused
    /// by name when the answer gives it a string instead, whether one label,
    /// a comma-separated list or empty. `parse` neither wraps nor splits it.
    #[test]
    fn an_array_field_given_a_string_is_refused_by_name() {
      let keys = keys_declared_as("array");
      assert_eq!(
        keys,
        [
          "subjects",
          "objects",
          "actions",
          "emotion",
          "lighting",
          "tags",
          "categories"
        ]
      );
      for key in keys {
        for value in [r#""label""#, r#""one, two; three""#, r#""""#] {
          assert_refused_by_name(key, value);
        }
      }
    }

    /// LAW: a field the schema declares as a string is refused by name when
    /// the answer gives it an array instead, a one-element array included.
    /// `parse` never unwraps it.
    #[test]
    fn a_string_field_given_an_array_is_refused_by_name() {
      let keys = keys_declared_as("string");
      assert_eq!(keys, ["scene", "description", "shot_type"]);
      for key in keys {
        for value in [r#"["label"]"#, "[]", r#"["one","two"]"#] {
          assert_refused_by_name(key, value);
        }
      }
    }

    /// LAW: `null` is refused by name for every field, an array field
    /// included: the schema allows `null` nowhere.
    #[test]
    fn a_null_field_is_refused_by_name() {
      for (key, _) in FULL_ANSWER {
        assert_refused_by_name(key, "null");
      }
    }

    /// LAW: a boolean or a number is refused by name for every field.
    #[test]
    fn a_boolean_or_number_field_is_refused_by_name() {
      for (key, _) in FULL_ANSWER {
        for value in ["true", "false", "0", "42", "-1.5"] {
          assert_refused_by_name(key, value);
        }
      }
    }

    /// LAW: an object is refused by name for every field: the schema
    /// declares no object-valued field.
    #[test]
    fn an_object_field_is_refused_by_name() {
      for (key, _) in FULL_ANSWER {
        for value in ["{}", r#"{"label":"office"}"#] {
          assert_refused_by_name(key, value);
        }
      }
    }

    /// LAW: an array field is refused by name when any element is not a
    /// string: `null`, a boolean, a number, an array or an object.
    #[test]
    fn an_array_field_with_an_element_that_is_not_a_string_is_refused_by_name() {
      for key in keys_declared_as("array") {
        for value in [
          r#"["label",null]"#,
          r#"["label",true]"#,
          r#"["label",42]"#,
          r#"["label",["nested"]]"#,
          r#"["label",{"label":"office"}]"#,
        ] {
          assert_refused_by_name(key, value);
        }
      }
    }

    // ===== every JSON text reaches the field checks =====

    /// JSON numbers a `serde_json::Value` cannot hold (beyond `f64`'s range
    /// either way, by exponent or by digits), numbers so small they round
    /// to zero, and the other spellings JSON's grammar admits. Each is a
    /// number, the wrong type for every field.
    fn numbers() -> Vec<String> {
      let mut numbers: Vec<String> = [
        "1e400",
        "-1e400",
        "1E400",
        "1e999999999999999999999",
        "123456789012345678901234567890e300",
        "1e-400",
        "-1e-400",
        "-0",
        "1E5",
        "1e+5",
        "0.5e-3",
      ]
      .into_iter()
      .map(String::from)
      .collect();
      numbers.push("9".repeat(400));
      numbers
    }

    /// LAW: a number is refused by name in every string field, whatever its
    /// magnitude or spelling: `1e400` is a JSON number as much as `1` is.
    #[test]
    fn a_number_of_any_magnitude_in_a_string_field_is_refused_by_name() {
      for key in keys_declared_as("string") {
        for number in numbers() {
          assert_refused_by_name(key, &number);
        }
      }
    }

    /// LAW: a number is refused by name in every array field, whatever its
    /// magnitude or spelling, as the field's value and as an element.
    #[test]
    fn a_number_of_any_magnitude_in_an_array_field_is_refused_by_name() {
      for key in keys_declared_as("array") {
        for number in numbers() {
          for value in [
            number.clone(),
            format!("[{number}]"),
            format!(r#"["label",{number}]"#),
          ] {
            assert_refused_by_name(key, &value);
          }
        }
      }
    }

    /// LAW: a number under a key the schema does not declare is
    /// `UnknownFields` naming the key, whatever its magnitude: as the
    /// member's value, nested inside it, and under a switched-off
    /// extension's key.
    #[test]
    fn a_number_of_any_magnitude_under_an_undeclared_key_is_unknown_fields() {
      let task = ImageAnalysisTask::new();
      for number in numbers() {
        for (key, value) in [
          ("extra", number.clone()),
          ("extra", format!(r#"{{"nested":[{number}]}}"#)),
          ("scene", number.clone()),
        ] {
          match task.parse(&format!(r#"{{{DEFAULT_MEMBERS},"{key}":{value}}}"#)) {
            Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, [key]),
            other => panic!("{key} = {value} must be UnknownFields naming {key}, got {other:?}"),
          }
        }
      }
    }

    /// LAW: a top-level value that is not an object names every field the
    /// task asks for, whatever numbers it holds.
    #[test]
    fn a_top_level_value_that_is_not_an_object_names_every_asked_for_field() {
      for (task, asked_for) in [
        (ImageAnalysisTask::new(), vec!["description", "tags"]),
        (full_task(), Field::ALL.map(Field::key).to_vec()),
      ] {
        for number in numbers() {
          for answer in [
            number.clone(),
            format!("[{number}]"),
            format!(r#"["label",{{"nested":{number}}}]"#),
          ] {
            match task.parse(&answer) {
              Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, asked_for),
              other => {
                panic!("{answer} must be MissingFields naming every asked-for field, got {other:?}")
              }
            }
          }
        }
      }
    }

    /// LAW: a duplicated key is `DuplicateField` whatever JSON values its
    /// copies hold.
    #[test]
    fn a_duplicated_key_is_named_whatever_its_copies_hold() {
      for (answer, repeated) in [
        (
          r#"{"description":1e400,"description":"a","tags":["label"]}"#,
          "description",
        ),
        (
          r#"{"extra":[[[1e400]]],"extra":1,"description":"a","tags":["label"]}"#,
          "extra",
        ),
      ] {
        match ImageAnalysisTask::new().parse(answer) {
          Err(JsonParseError::DuplicateField(key)) => assert_eq!(key, repeated),
          other => panic!("{answer} must be DuplicateField naming {repeated}, got {other:?}"),
        }
      }
    }

    /// How deep the nesting law below nests: far past serde_json's own
    /// recursion limit of 128, and deep enough that a reader recursing once
    /// per level would exhaust a test thread's stack.
    const DEPTH: usize = 100_000;

    /// LAW: a value nested to any depth is a wrong type like any other. In
    /// every field, as the value or as an array field's element, it is
    /// `MissingFields` naming the field; under an undeclared key it is
    /// `UnknownFields`; as the whole answer, an array names every
    /// asked-for field and an object its undeclared key.
    #[test]
    fn a_value_nested_to_any_depth_is_refused_by_name() {
      let nested_array = format!("{}{}", "[".repeat(DEPTH), "]".repeat(DEPTH));
      let nested_object = format!("{}null{}", r#"{"a":"#.repeat(DEPTH), "}".repeat(DEPTH));
      let array_keys = keys_declared_as("array");
      for (shape, nested) in [("array", &nested_array), ("object", &nested_object)] {
        for (key, _) in FULL_ANSWER {
          let mut values = vec![nested.clone()];
          if array_keys.contains(&key) {
            values.push(format!(r#"["label",{nested}]"#));
          }
          for value in values {
            match full_task().parse(&full_answer_with(key, &value)) {
              Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, [key]),
              other => panic!(
                "{key}: an {shape} nested {DEPTH} deep must be MissingFields naming {key}, got {other:?}"
              ),
            }
          }
        }
        match ImageAnalysisTask::new().parse(&format!(r#"{{{DEFAULT_MEMBERS},"extra":{nested}}}"#))
        {
          Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, ["extra"]),
          other => panic!(
            "an {shape} nested {DEPTH} deep under an undeclared key must be UnknownFields, got {other:?}"
          ),
        }
      }
      match ImageAnalysisTask::new().parse(&nested_array) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["description", "tags"]),
        other => panic!(
          "an array nested {DEPTH} deep as the answer must name every asked-for field, got {other:?}"
        ),
      }
      match ImageAnalysisTask::new().parse(&nested_object) {
        Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, ["a"]),
        other => panic!(
          "an object nested {DEPTH} deep as the answer must name its undeclared key, got {other:?}"
        ),
      }
    }

    /// LAW: a string of any length reaches the field checks: a description
    /// over its cap is refused by name, a long string under an undeclared
    /// key is `UnknownFields`, and a long label, which no cap limits,
    /// parses whole.
    #[test]
    fn a_string_of_any_length_reaches_the_field_checks() {
      let long = "a".repeat(1_000_000);
      let task = ImageAnalysisTask::new();
      match task.parse(&format!(r#"{{"description":"{long}","tags":["label"]}}"#)) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["description"]),
        other => panic!(
          "a long description must be MissingFields, got {:?}",
          other.map(|_| ())
        ),
      }
      match task.parse(&format!(r#"{{{DEFAULT_MEMBERS},"extra":"{long}"}}"#)) {
        Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, ["extra"]),
        other => panic!(
          "a long string under an undeclared key must be UnknownFields, got {:?}",
          other.map(|_| ())
        ),
      }
      let analysis = task
        .parse(&format!(
          r#"{{"description":"A person reads.","tags":["{long}"]}}"#
        ))
        .expect("a label of any length parses");
      assert!(analysis.tags() == [SmolStr::from(long.as_str())]);
    }

    /// LAW: the number spellings JSON's grammar admits reach the field
    /// checks and are refused by the field's name; the spellings it does
    /// not admit are not JSON.
    #[test]
    fn only_the_number_spellings_json_admits_reach_the_field_checks() {
      let task = ImageAnalysisTask::new();
      let answer = |number: &str| format!(r#"{{"description":{number},"tags":["label"]}}"#);
      for number in ["-0", "1E5", "1e+5", "0.5e-3", "-0.0e-0", "1e400"] {
        match task.parse(&answer(number)) {
          Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["description"]),
          other => panic!("{number} is a JSON number, refused by name; got {other:?}"),
        }
      }
      for spelling in [
        "01",
        "+1",
        ".5",
        "1.",
        "1e",
        "1e+",
        "-",
        "--1",
        "0x10",
        "NaN",
        "Infinity",
        "-Infinity",
      ] {
        assert!(
          matches!(task.parse(&answer(spelling)), Err(JsonParseError::Json(_))),
          "{spelling} is not a JSON number"
        );
      }
    }

    /// Strings that escape one half of a UTF-16 surrogate pair without the
    /// other, as JSON string literals: a leading half alone, at the end and
    /// before text; a trailing half alone; two leading halves; the halves
    /// in the wrong order; a trailing half after a whole pair; and a
    /// leading half before another escape. JSON's grammar admits each; no
    /// Unicode text can hold one.
    const LONE_SURROGATES: [&str; 7] = [
      r#""\uD800""#,
      r#""\uDBFFb""#,
      r#""\uDC00""#,
      r#""\uD800\uD800""#,
      r#""\uDFFF\uD800""#,
      r#""\uD83D\uDE00\uDC00""#,
      r#""a\uD800\n""#,
    ];

    /// LAW: a string escaping half a surrogate pair is not JSON, wherever
    /// it sits: in a string field, in an array field's element, inside a
    /// value of the wrong type, under an undeclared key, nested there, as a
    /// key, and in a top-level value that is not an object. `parse` decodes
    /// few of those strings and refuses every one as `Json`, positioned at
    /// the escape.
    #[test]
    fn a_string_escaping_half_a_surrogate_pair_is_not_json_wherever_it_sits() {
      let task = ImageAnalysisTask::new();
      for lone in LONE_SURROGATES {
        for answer in [
          format!(r#"{{"description":{lone},"tags":["label"]}}"#),
          format!(r#"{{"description":"A person reads.","tags":["label",{lone}]}}"#),
          format!(r#"{{"description":[{lone}],"tags":["label"]}}"#),
          format!(r#"{{"description":"A person reads.","tags":[{{"label":{lone}}}]}}"#),
          format!(r#"{{{DEFAULT_MEMBERS},"extra":{lone}}}"#),
          format!(r#"{{{DEFAULT_MEMBERS},"extra":{{"nested":[{lone}]}}}}"#),
          format!(r#"{{{DEFAULT_MEMBERS},{lone}:"label"}}"#),
          format!("[{lone}]"),
        ] {
          assert!(
            matches!(task.parse(&answer), Err(JsonParseError::Json(_))),
            "{answer} is not JSON"
          );
        }
      }
      let answer = "{\"description\":\"A person reads.\",\n\"tags\":[\"\\uD800\"]}";
      match task.parse(answer) {
        Err(JsonParseError::Json(err)) => assert_eq!((err.line(), err.column()), (2, 10)),
        other => panic!("{answer} must be Json at the escape, got {other:?}"),
      }
    }

    /// LAW: a whole surrogate pair is text, and so is an escaped backslash
    /// before `u`: the pair decodes to its one character, and `\\uD800` to
    /// a backslash and the letters `uD800`.
    #[test]
    fn a_whole_surrogate_pair_and_an_escaped_backslash_are_text() {
      let analysis = ImageAnalysisTask::new()
        .parse(r#"{"description":"\uD83D\uDE00 A person smiles.","tags":["\\uD800"]}"#)
        .expect("a surrogate pair and an escaped backslash are text");
      assert_eq!(analysis.description(), "\u{1F600} A person smiles.");
      assert_eq!(analysis.tags(), &[SmolStr::from(r"\uD800")][..]);
    }

    /// LAW: `find_lone_surrogate` flags exactly the string literals
    /// serde_json cannot decode, and points at an escape when it flags one,
    /// over every sequence of up to four pieces drawn from `\u` escapes on
    /// each side of every surrogate boundary, the two-byte escapes, text
    /// that spells a `\u` escape after an escaped backslash, and other
    /// text.
    #[test]
    fn the_lone_surrogate_check_agrees_with_serde_json() {
      const PIECES: [&str; 13] = [
        r"\u0041", r"\uD7FF", r"\uD800", r"\uDBFF", r"\uDC00", r"\uDFFF", r"\uE000", r"\\",
        r#"\""#, r"\n", r"\/", "uDC00", "é",
      ];
      let mut literals = Vec::new();
      let mut bodies = vec![String::new()];
      for length in 0..=4 {
        literals.extend(bodies.iter().map(|body| format!(r#""{body}""#)));
        if length < 4 {
          bodies = bodies
            .iter()
            .flat_map(|body| PIECES.map(|piece| format!("{body}{piece}")))
            .collect();
        }
      }
      assert_eq!(literals.len(), 1 + 13 + 169 + 2197 + 28561);
      for literal in &literals {
        let decodes = serde_json::from_str::<String>(literal).is_ok();
        let found = find_lone_surrogate(literal);
        assert_eq!(
          found.is_none(),
          decodes,
          "{literal}: serde_json decodes it: {decodes}; found {found:?}"
        );
        if let Some(escape) = found {
          assert_eq!(literal.as_bytes()[escape], b'\\', "{literal}: {escape}");
        }
      }
    }

    /// LAW: an answer that is not JSON is `Json` before any other refusal.
    /// A duplicated key, an undeclared key or a wrong type earlier in the
    /// text does not stand in for the text not being JSON.
    #[test]
    fn an_answer_that_is_not_json_is_json_before_any_other_refusal() {
      let task = ImageAnalysisTask::new();
      for answer in [
        r#"{"description":"a","description":"b","tags":["label"],}"#,
        r#"{"description":"a","description":"b","tags":["\uD800"]}"#,
        r#"{"extra":1e400,"description":"a","tags":["label"]"#,
        r#"{"description":1e400,"tags":["label"]} trailing"#,
      ] {
        assert!(
          matches!(task.parse(answer), Err(JsonParseError::Json(_))),
          "{answer} is not JSON"
        );
      }
    }
  }
}
