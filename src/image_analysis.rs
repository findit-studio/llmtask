//! `ImageAnalysis` — the canonical single-image VLM output type — and
//! [`ImageAnalysisTask`] (behind the `json` feature) — the canonical
//! `Task` implementation that produces it: prompt, JSON Schema, and a
//! resilient parser tolerant of constrained-decoder output drift.
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

  use serde::de::{self, Deserializer as _, MapAccess, SeqAccess, Visitor};
  use serde_json::{Map, Value, json};
  use smol_str::SmolStr;
  // Bring `String` into scope under both std (resolves via the
  // `extern crate std`) and alloc-only (resolves via the
  // `extern crate alloc as std` alias in lib.rs) — needed by the prompt
  // builder and `TopLevelVisitor::visit_string`.
  use std::{string::String, vec::Vec};

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

  /// Separators `tags`' comma-separated string form is split on (see
  /// [`extract_tags`]).
  const TAG_SEPARATORS: [char; 3] = [',', ';', '\n'];

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
  /// enforces them; `parse` checks them again and refuses an answer over
  /// either cap as [`JsonParseError::MissingFields`] naming the field. It
  /// never truncates or rewrites the model's words.
  ///
  /// # Example
  ///
  /// ```
  /// use llmtask::{
  ///   Task,
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
    /// time, but `parse` cannot inherit for free: a `serde_json::Map`
    /// decode of already-generated text accepts any key regardless of what
    /// the schema says, so `parse` enforces this promise itself via
    /// `unknown_fields`, and the caps via `unusable_fields`.
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

    /// `field`'s entry in the schema's `properties`. `description` and
    /// `tags` carry the task's caps.
    fn property_schema(&self, field: Field) -> Value {
      match field {
        Field::Description => json!({
            "type": "string",
            "maxLength": self.description_max_chars.get()
        }),
        Field::Tags => json!({
            "type": "array",
            "items": { "type": "string" },
            "maxItems": self.tags_max_items.get()
        }),
        Field::Scene | Field::ShotType => json!({ "type": "string" }),
        Field::Subjects
        | Field::Objects
        | Field::Actions
        | Field::Emotion
        | Field::Lighting
        | Field::Categories => json!({ "type": "array", "items": { "type": "string" } }),
      }
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

    /// Names the keys in `object` that the task's schema does not declare
    /// — any key outside the fields the task asks for, a switched-off
    /// extension's field included. The runtime enforcement of the schema's
    /// `additionalProperties: false` (see [`Self::build_schema`]).
    ///
    /// Owned [`SmolStr`] rather than `&'static str`: unlike a declared
    /// field name, an unknown key isn't known at compile time — it's
    /// whatever text the decoder emitted.
    fn unknown_fields(&self, object: &Map<String, Value>) -> Vec<SmolStr> {
      object
        .keys()
        .filter(|key| !self.fields().any(|field| field.key() == key.as_str()))
        .map(SmolStr::new)
        .collect()
    }

    /// Names the fields the task asks for that `object` leaves unusable:
    /// absent or JSON `null`, present with a JSON type the field's shape
    /// can't hold (e.g. a number where a string or array of strings is
    /// expected), or over a cap the schema declares (see
    /// [`Self::exceeds_cap`]). Keys outside the roster are a separate
    /// concern, handled by [`Self::unknown_fields`].
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
    /// model's words to fit. Every unusable field is caught in this one
    /// pass and named.
    fn unusable_fields(&self, object: &Map<String, Value>) -> Vec<&'static str> {
      self
        .fields()
        .filter(|&field| match object.get(field.key()) {
          None | Some(Value::Null) => true,
          Some(value) => !field_is_well_shaped(field, value) || self.exceeds_cap(field, value),
        })
        .map(Field::key)
        .collect()
    }

    /// `true` iff `value` holds more than the cap `field`'s schema entry
    /// declares, counted the way the schema counts: `description` in
    /// Unicode scalar values of the string as the answer wrote it (before
    /// `parse` trims it), `tags` in elements of the array. `tags`'
    /// comma-separated string form — drift `parse` tolerates, see
    /// [`extract_tags`] — counts the non-empty labels it lists, duplicates
    /// included, as the array form would.
    fn exceeds_cap(&self, field: Field, value: &Value) -> bool {
      match (field, value) {
        (Field::Description, Value::String(description)) => {
          description.chars().count() > self.description_max_chars.get()
        }
        (Field::Tags, Value::Array(items)) => items.len() > self.tags_max_items.get(),
        (Field::Tags, Value::String(list)) => {
          tag_segments(list).count() > self.tags_max_items.get()
        }
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
      // Not a plain `serde_json::from_str` (see `parse_top_level_value`'s
      // doc comment): that collapses a duplicate top-level member — later
      // overwrites earlier — before any check below ever sees both copies.
      let value: Value = parse_top_level_value(raw.trim())?;
      let Some(object) = value.as_object() else {
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
      // constrained-decoding engine, but nothing about decoding the
      // already-generated text through `serde_json::Map` — which accepts
      // any key — enforces it. Checked before `unusable_fields` so an
      // object with both an unknown key and a missing/invalid declared
      // field is named for the unknown key first (a structural violation
      // of "which keys are even allowed" takes precedence over per-field
      // shape checks).
      let unknown = self.unknown_fields(object);
      if !unknown.is_empty() {
        return Err(JsonParseError::UnknownFields(unknown));
      }
      let unusable = self.unusable_fields(object);
      if !unusable.is_empty() {
        return Err(JsonParseError::MissingFields(unusable));
      }
      // Every field `unusable_fields` didn't flag is now known to carry
      // a JSON shape its `extract_*` helper can consume, so extraction
      // itself is infallible from here. A field the task does not ask for
      // is never read and keeps its empty default.
      let mut result = ImageAnalysis::new();
      for field in self.fields() {
        let key = field.key();
        match field {
          Field::Scene => result.set_scene(extract_label(object, key)),
          Field::Description => result.set_description(extract_label(object, key)),
          Field::Subjects => result.set_subjects(extract_detection_array(object, key)),
          Field::Objects => result.set_objects(extract_detection_array(object, key)),
          Field::Actions => result.set_actions(extract_detection_array(object, key)),
          Field::Emotion => result.set_emotion(extract_detection_array(object, key)),
          Field::ShotType => result.set_shot_type(extract_shot_type(object)),
          Field::Lighting => result.set_lighting(extract_detection_array(object, key)),
          Field::Tags => result.set_tags(extract_tags(object)),
          Field::Categories => result.set_categories(extract_detection_array(object, key)),
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

  // ===== duplicate-checked top-level parse (Codex R2, PR #5) =====

  /// Deserializes `raw` into a [`Value`], refusing a TOP-level object
  /// member name that appears more than once, instead of silently
  /// collapsing it the way `serde_json::from_str::<Value>` (what this
  /// replaced) does.
  ///
  /// `from_str::<Value>` builds the object via repeated `Map::insert`, so
  /// a later member overwrites an earlier one with no trace left behind:
  /// `{"categories": null, "categories": []}` and `{"categories": []}`
  /// deserialize to the identical `Value`, while the reverse key order
  /// (`{"categories": [], "categories": null}`) deserializes to the same
  /// `Value` as `{"categories": null}` alone. Both orderings are schema
  /// violations no compliant decoder should emit, but only one of the two
  /// used to be caught — by `unusable_fields`, and only because `null`
  /// happened to survive the collapse — while the reverse order silently
  /// passed; a duplicated key **outside** the declared fields could never
  /// be named by `unknown_fields` at all, because by the time it runs the
  /// `Map` remembers only the surviving copy. Checking here, before any
  /// `Value` is built, is the only point where both copies are still
  /// visible to compare.
  ///
  /// Scope is TOP-level only, deliberately: every property a task's schema
  /// can declare is `string` or `array of string` — this contract has
  /// no `object`-valued field, at the top level or nested. A JSON object
  /// can therefore only legitimately appear here as the document root;
  /// anywhere else (e.g. an object smuggled into a `subjects` array
  /// element in place of a string) it's already a wrong-shaped value that
  /// [`field_is_well_shaped`] rejects on type alone, regardless of what
  /// its own internal keys collapsed to. A future field that legitimately
  /// nests an object would need this same duplicate check extended to it.
  ///
  /// This is also the crate's only JSON entry point: [`ImageAnalysisTask::parse`]
  /// calls nothing else that builds a `Value` from text — no fenced-code
  /// stripping, no secondary lenient parse. `reject_fenced_json` and
  /// `reject_json_with_wrapper_text` (in the tests below) pass because
  /// `raw.trim()` isn't valid/complete JSON on its own, not via a
  /// different code path, so routing through here covers the whole parse
  /// surface — there is no second `from_str`/`from_value` call anywhere in
  /// this crate for a fenced or prose-wrapped variant to bypass.
  fn parse_top_level_value(raw: &str) -> Result<Value, JsonParseError> {
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    let duplicate: RefCell<Option<SmolStr>> = RefCell::new(None);
    let visited = deserializer.deserialize_any(TopLevelVisitor {
      duplicate: &duplicate,
    });
    match visited {
      Ok(value) => {
        // Mirrors `serde_json::from_str`'s own trailing-content check:
        // rejects prose after the JSON object, an unclosed fence, etc.
        // (`reject_fenced_json`, `reject_json_with_wrapper_text`).
        deserializer.end()?;
        Ok(value)
      }
      Err(err) => match duplicate.into_inner() {
        Some(key) => Err(JsonParseError::DuplicateField(key)),
        None => Err(JsonParseError::Json(err)),
      },
    }
  }

  /// [`Visitor`] behind [`parse_top_level_value`]. Mirrors
  /// `serde_json::Value`'s own `Deserialize` impl method-for-method —
  /// every shape produces the identical `Value` — except [`Self::visit_map`],
  /// which refuses a repeated key instead of overwriting the earlier entry.
  ///
  /// `visit_i128` / `visit_u128` / `visit_none` / `visit_some` are not
  /// overridden: this crate doesn't enable serde_json's
  /// `arbitrary_precision` feature, and without it `deserialize_any` only
  /// ever calls `visit_f64` / `visit_u64` / `visit_i64` for a JSON number
  /// and `visit_unit` for `null` (verified against `serde_json`'s own
  /// `ParserNumber` and null-literal dispatch). Those four methods are
  /// unreachable through `serde_json::Deserializer`, so overriding them
  /// here would be untested dead code; the `Visitor` trait's default
  /// implementations (which forward sensibly on their own) still apply if
  /// that ever changes.
  struct TopLevelVisitor<'a> {
    duplicate: &'a RefCell<Option<SmolStr>>,
  }

  impl<'de> Visitor<'de> for TopLevelVisitor<'_> {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
      Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
      Ok(Value::from(v))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
      Ok(Value::from(v))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
      Ok(Value::from(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E>
    where
      E: de::Error,
    {
      self.visit_string(String::from(v))
    }

    fn visit_string<E>(self, v: String) -> Result<Value, E> {
      Ok(Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
      Ok(Value::Null)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Value, A::Error>
    where
      A: SeqAccess<'de>,
    {
      let mut vec = Vec::new();
      while let Some(elem) = seq.next_element()? {
        vec.push(elem);
      }
      Ok(Value::Array(vec))
    }

    /// The one method that differs from stock `Value` decoding: a member
    /// name seen earlier in this same object is refused. `MapAccess`'s
    /// error type is fixed to `serde_json::Error` by the driving
    /// `Deserializer`, which has no variant that names an arbitrary
    /// field, so the key travels out through `self.duplicate` instead —
    /// [`parse_top_level_value`] reads it back after the `Err` this
    /// returns propagates up through `deserialize_any`.
    fn visit_map<A>(self, mut map: A) -> Result<Value, A::Error>
    where
      A: MapAccess<'de>,
    {
      let mut object = Map::new();
      while let Some(key) = map.next_key::<String>()? {
        if object.contains_key(&key) {
          *self.duplicate.borrow_mut() = Some(SmolStr::new(&key));
          return Err(de::Error::custom("duplicate top-level key"));
        }
        let value: Value = map.next_value()?;
        object.insert(key, value);
      }
      Ok(Value::Object(object))
    }
  }

  /// `true` iff `value`'s JSON type is one `field`'s extractor can
  /// consume. Presence/null is checked separately by the caller
  /// (`unusable_fields` treats an absent or `Value::Null` field as
  /// unusable without consulting this function); this only judges the
  /// shape of a value that's actually present and non-null.
  fn field_is_well_shaped(field: Field, value: &Value) -> bool {
    match field {
      // `scene` / `description`: bare string only — no array-wrapping
      // tolerance (unlike `shot_type` below).
      Field::Scene | Field::Description => value.is_string(),
      // `shot_type`: bare string, or the sole element of a one-element
      // array (tolerates a constrained decoder that wraps a scalar).
      Field::ShotType => match value {
        Value::String(_) => true,
        Value::Array(items) => items.len() == 1 && items[0].is_string(),
        _ => false,
      },
      // Every array-shaped field: a JSON array of strings, or a bare
      // string tolerated as a single-element array.
      Field::Subjects
      | Field::Objects
      | Field::Actions
      | Field::Emotion
      | Field::Lighting
      | Field::Tags
      | Field::Categories => match value {
        Value::String(_) => true,
        Value::Array(items) => items.iter().all(Value::is_string),
        _ => false,
      },
    }
  }

  /// Extracts the trimmed string at `field` (`scene` / `description`),
  /// or an empty `SmolStr` when absent/null. Caller must have already
  /// confirmed via `unusable_fields` that a present value is a JSON
  /// string.
  fn extract_label(object: &Map<String, Value>, field: &str) -> SmolStr {
    match object.get(field) {
      Some(Value::String(s)) => SmolStr::new(s.trim()),
      _ => SmolStr::default(),
    }
  }

  /// Extracts `shot_type`: a bare string, or the sole element of a
  /// one-element array. Caller must have already confirmed the shape.
  fn extract_shot_type(object: &Map<String, Value>) -> SmolStr {
    match object.get("shot_type") {
      Some(Value::String(s)) => SmolStr::new(s.trim()),
      Some(Value::Array(items)) => match items.first() {
        Some(Value::String(s)) => SmolStr::new(s.trim()),
        _ => SmolStr::default(),
      },
      _ => SmolStr::default(),
    }
  }

  /// Extracts a detection-style label array (`subjects`, `objects`,
  /// `actions`, `emotion`, `lighting`, `categories`): a JSON array
  /// trims and dedupes each element verbatim — no comma-splitting,
  /// because detection labels can themselves contain commas (e.g. "red,
  /// white, and blue flag") — or a single JSON string wraps as one
  /// label. Absent/null yields an empty list. Caller must have already
  /// confirmed the shape via `unusable_fields`.
  fn extract_detection_array(object: &Map<String, Value>, field: &str) -> Vec<SmolStr> {
    let mut values = Vec::new();
    match object.get(field) {
      Some(Value::String(s)) => push_label(&mut values, s),
      Some(Value::Array(items)) => {
        for item in items {
          if let Value::String(s) = item {
            push_label(&mut values, s);
          }
        }
      }
      _ => {}
    }
    values
  }

  /// Extracts `tags`: like [`extract_detection_array`], but a JSON
  /// string form is additionally split on [`TAG_SEPARATORS`] (see
  /// [`tag_segments`]) — tag-list drift (model dropped the array around a
  /// flat comma-separated string) is the historically common case for this
  /// specific field. The array form is never split (a tag like "july 4,
  /// 2026" must stay one entry).
  fn extract_tags(object: &Map<String, Value>) -> Vec<SmolStr> {
    let mut values = Vec::new();
    match object.get("tags") {
      Some(Value::String(s)) => {
        for part in tag_segments(s) {
          push_label(&mut values, part);
        }
      }
      Some(Value::Array(items)) => {
        for item in items {
          if let Value::String(s) = item {
            push_label(&mut values, s);
          }
        }
      }
      _ => {}
    }
    values
  }

  /// The labels `tags`' comma-separated string form lists: the trimmed,
  /// non-empty parts between [`TAG_SEPARATORS`]. `extract_tags` reads
  /// them and `ImageAnalysisTask::exceeds_cap` counts them, so both
  /// agree on what one label is.
  fn tag_segments(list: &str) -> impl Iterator<Item = &str> {
    list
      .split(TAG_SEPARATORS)
      .map(str::trim)
      .filter(|part| !part.is_empty())
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
    /// `raw.trim()` to already be a bare JSON object, and does not strip
    /// wrapping of any kind (fences included).
    #[test]
    fn reject_fenced_json() {
      let text = "```json\n{\"scene\":\"office\",\"description\":\"People working\"}\n```";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    /// `tags` is always asked for, so the comma-separated tolerance holds
    /// on the default task.
    #[test]
    fn parse_comma_separated_tag_string() {
      let json = r#"{"description":"A singer on stage","tags":"concert, live music, spotlight"}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(
        result.tags(),
        &[
          SmolStr::from("concert"),
          SmolStr::from("live music"),
          SmolStr::from("spotlight"),
        ][..]
      );
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
    /// Before this fix, collapse kept the *null* second copy, so
    /// `unusable_fields` already rejected this order (as
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
    /// this fix, collapse kept the *wrong-typed* second copy, so
    /// `unusable_fields` already rejected this order (as
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

    #[test]
    fn subjects_string_form_treated_as_single_label() {
      let json = r#"{"scene":"x","description":"y","subjects":"middle-aged man, in red jacket","objects":[],"actions":[],"emotion":[],"shot_type":"x","lighting":[],"tags":["t"],"categories":[]}"#;
      let task = full_task();
      let result = task.parse(json).expect("string-form parse");
      assert_eq!(
        result.subjects().len(),
        1,
        "string-form must wrap as a single label, not comma-split"
      );
      assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
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

    #[test]
    fn parse_shot_type_list_form() {
      // shot_type accepts the list form `["wide shot"]` (one element).
      let json_one = r#"{"description":"y","shot_type":["wide shot"],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new().with_extensions([Extension::ShotType]);
      let result = task.parse(json_one).expect("single-element list parse");
      assert_eq!(result.shot_type(), "wide shot");
    }

    /// A multi-element `shot_type` array is now a named
    /// `MissingFields(["shot_type"])` error rather than the generic
    /// serde message the two engine copies produced ("expected a
    /// single shot_type label, got multiple values", wrapped as
    /// `JsonParseError::Json`) — folded into `unusable_fields` like
    /// every other shape violation.
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

    /// Codex R1 (PR #5): `unusable_fields` used to exempt a *present*
    /// `categories: null` from the shape check (it fell through a guard
    /// that only made sense back when a bare `!matches!(v, Value::Null)`
    /// meant "is this key even here" — categories was still optional at
    /// the time), silently defaulting it to an empty list even though
    /// the schema's `categories` entry allows only an array of strings,
    /// never null. Now that `categories` is a required field (see
    /// `categories_absent_is_rejected` above), a present `null` and a
    /// totally absent key both fall through the exact same
    /// `None | Some(Value::Null) => false` arm in `unusable_fields` and
    /// produce the same named error — this test keeps the present-null
    /// input shape pinned separately from the absent-key shape so a
    /// future regression in either branch is still caught.
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
    /// finding): `field_is_well_shaped`'s array arm checks
    /// `items.iter().all(Value::is_string)`, enforcing the schema's
    /// `items: {"type": "string"}` promise for every array-shaped field.
    /// That was already implemented but had no element-level regression
    /// — only a whole-field wrong type (`subjects: 42`, above) had
    /// coverage, which exercises a different branch of
    /// `field_is_well_shaped` than a well-typed array with one bad
    /// element does.
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
    /// wrote it; `tags` in the labels it lists, in either form.
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
      let analysis = task
        .parse(r#"{"description":"a","tags":"x, y, z,"}"#)
        .expect("a trailing separator lists no fourth tag");
      assert_eq!(analysis.tags().len(), 3);
      assert_eq!(
        refusal(r#"{"description":"a","tags":["x","y","z","w"]}"#),
        ["tags"]
      );
      assert_eq!(
        refusal(r#"{"description":"a","tags":"x, y, z, w"}"#),
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
  }
}
