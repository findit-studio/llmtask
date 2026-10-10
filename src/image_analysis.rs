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

// The account a description is settled by. It is defined beside the
// `Task::parse_ended` door that hands it to any task, and stays nameable
// here, beside the `DescriptionEnd` it settles.
pub use crate::task::FieldEnd;

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
///
/// With the `serde` feature the fields serialize in the order declared: the
/// ten fields 0.4.x wrote, then [`description_end`](Self::description_end),
/// then the six list ends from [`subjects_end`](Self::subjects_end) to
/// [`categories_end`](Self::categories_end), each mark after every field
/// written before it existed. A self-describing document written before a
/// mark existed (JSON from 0.4.x, or from 0.5.x for the list ends) reads it
/// as unknown ([`DescriptionEnd::Unknown`], [`ListEnd::Unknown`]), the honest
/// value for an end nothing recorded. A positional format (bincode) cannot
/// tell an absent field from a present one, so a payload written before a
/// mark existed does not read in one.
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
  // After the ten fields 0.4.x wrote, so they keep their places.
  #[cfg_attr(feature = "serde", serde(default))]
  description_end: DescriptionEnd,
  // After every field 0.5.x wrote, so those keep their places.
  #[cfg_attr(feature = "serde", serde(default))]
  subjects_end: ListEnd,
  #[cfg_attr(feature = "serde", serde(default))]
  objects_end: ListEnd,
  #[cfg_attr(feature = "serde", serde(default))]
  actions_end: ListEnd,
  #[cfg_attr(feature = "serde", serde(default))]
  emotion_end: ListEnd,
  #[cfg_attr(feature = "serde", serde(default))]
  lighting_end: ListEnd,
  #[cfg_attr(feature = "serde", serde(default))]
  categories_end: ListEnd,
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
  ///
  /// Resets [`description_end`](Self::description_end) to
  /// [`DescriptionEnd::Unknown`], as [`Self::set_description`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_description(mut self, val: impl Into<SmolStr>) -> Self {
    self.set_description(val);
    self
  }

  /// In-place setter for `description`. Pass an empty string to clear.
  ///
  /// Resets [`description_end`](Self::description_end) to
  /// [`DescriptionEnd::Unknown`]: how the old text ended says nothing about
  /// the new one, so a caller holding the decoder's account of the new
  /// text sets it afterwards.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_description(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.description = val.into();
    self.description_end = DescriptionEnd::Unknown;
    self
  }

  // --- description_end ---

  /// How the description ends: where its writer ended it, or ragged where
  /// a length cap stopped it — or unknown. See [`DescriptionEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn description_end(&self) -> DescriptionEnd {
    self.description_end
  }

  /// Builder-style setter for `description_end`. Set it after the
  /// description: setting the description resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_description_end(mut self, val: DescriptionEnd) -> Self {
    self.description_end = val;
    self
  }

  /// In-place setter for `description_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_description_end(&mut self, val: DescriptionEnd) -> &mut Self {
    self.description_end = val;
    self
  }

  // --- subjects ---

  /// Distinct people or animals visible in the scene.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn subjects(&self) -> &[SmolStr] {
    &self.subjects
  }

  /// Builder-style setter for `subjects`.
  ///
  /// Resets [`subjects_end`](Self::subjects_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_subjects`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_subjects(mut self, val: Vec<SmolStr>) -> Self {
    self.set_subjects(val);
    self
  }

  /// In-place setter for `subjects`.
  ///
  /// Resets [`subjects_end`](Self::subjects_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_subjects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.subjects = val;
    self.subjects_end = ListEnd::Unknown;
    self
  }

  // --- subjects_end ---

  /// How `subjects` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn subjects_end(&self) -> ListEnd {
    self.subjects_end
  }

  /// Builder-style setter for `subjects_end`. Set it after `subjects`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_subjects_end(mut self, val: ListEnd) -> Self {
    self.subjects_end = val;
    self
  }

  /// In-place setter for `subjects_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_subjects_end(&mut self, val: ListEnd) -> &mut Self {
    self.subjects_end = val;
    self
  }

  // --- objects ---

  /// Notable, search-relevant objects.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn objects(&self) -> &[SmolStr] {
    &self.objects
  }

  /// Builder-style setter for `objects`.
  ///
  /// Resets [`objects_end`](Self::objects_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_objects`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_objects(mut self, val: Vec<SmolStr>) -> Self {
    self.set_objects(val);
    self
  }

  /// In-place setter for `objects`.
  ///
  /// Resets [`objects_end`](Self::objects_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_objects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.objects = val;
    self.objects_end = ListEnd::Unknown;
    self
  }

  // --- objects_end ---

  /// How `objects` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn objects_end(&self) -> ListEnd {
    self.objects_end
  }

  /// Builder-style setter for `objects_end`. Set it after `objects`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_objects_end(mut self, val: ListEnd) -> Self {
    self.objects_end = val;
    self
  }

  /// In-place setter for `objects_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_objects_end(&mut self, val: ListEnd) -> &mut Self {
    self.objects_end = val;
    self
  }

  // --- actions ---

  /// Visible actions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn actions(&self) -> &[SmolStr] {
    &self.actions
  }

  /// Builder-style setter for `actions`.
  ///
  /// Resets [`actions_end`](Self::actions_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_actions`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_actions(mut self, val: Vec<SmolStr>) -> Self {
    self.set_actions(val);
    self
  }

  /// In-place setter for `actions`.
  ///
  /// Resets [`actions_end`](Self::actions_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_actions(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.actions = val;
    self.actions_end = ListEnd::Unknown;
    self
  }

  // --- actions_end ---

  /// How `actions` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn actions_end(&self) -> ListEnd {
    self.actions_end
  }

  /// Builder-style setter for `actions_end`. Set it after `actions`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_actions_end(mut self, val: ListEnd) -> Self {
    self.actions_end = val;
    self
  }

  /// In-place setter for `actions_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_actions_end(&mut self, val: ListEnd) -> &mut Self {
    self.actions_end = val;
    self
  }

  // --- emotion ---

  /// Scene-level emotion terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn emotion(&self) -> &[SmolStr] {
    &self.emotion
  }

  /// Builder-style setter for `emotion`.
  ///
  /// Resets [`emotion_end`](Self::emotion_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_emotion`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_emotion(mut self, val: Vec<SmolStr>) -> Self {
    self.set_emotion(val);
    self
  }

  /// In-place setter for `emotion`.
  ///
  /// Resets [`emotion_end`](Self::emotion_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_emotion(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.emotion = val;
    self.emotion_end = ListEnd::Unknown;
    self
  }

  // --- emotion_end ---

  /// How `emotion` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn emotion_end(&self) -> ListEnd {
    self.emotion_end
  }

  /// Builder-style setter for `emotion_end`. Set it after `emotion`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_emotion_end(mut self, val: ListEnd) -> Self {
    self.emotion_end = val;
    self
  }

  /// In-place setter for `emotion_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_emotion_end(&mut self, val: ListEnd) -> &mut Self {
    self.emotion_end = val;
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
  ///
  /// Resets [`lighting_end`](Self::lighting_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_lighting`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_lighting(mut self, val: Vec<SmolStr>) -> Self {
    self.set_lighting(val);
    self
  }

  /// In-place setter for `lighting`.
  ///
  /// Resets [`lighting_end`](Self::lighting_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_lighting(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.lighting = val;
    self.lighting_end = ListEnd::Unknown;
    self
  }

  // --- lighting_end ---

  /// How `lighting` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn lighting_end(&self) -> ListEnd {
    self.lighting_end
  }

  /// Builder-style setter for `lighting_end`. Set it after `lighting`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_lighting_end(mut self, val: ListEnd) -> Self {
    self.lighting_end = val;
    self
  }

  /// In-place setter for `lighting_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_lighting_end(&mut self, val: ListEnd) -> &mut Self {
    self.lighting_end = val;
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
  ///
  /// Resets [`categories_end`](Self::categories_end) to [`ListEnd::Unknown`], as
  /// [`Self::set_categories`] does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_categories(mut self, val: Vec<SmolStr>) -> Self {
    self.set_categories(val);
    self
  }

  /// In-place setter for `categories`.
  ///
  /// Resets [`categories_end`](Self::categories_end) to [`ListEnd::Unknown`]: how the
  /// old list ended says nothing about the new one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_categories(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.categories = val;
    self.categories_end = ListEnd::Unknown;
    self
  }

  // --- categories_end ---

  /// How `categories` ends: closed short of its cap, or capped — or unknown.
  /// See [`ListEnd`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn categories_end(&self) -> ListEnd {
    self.categories_end
  }

  /// Builder-style setter for `categories_end`. Set it after `categories`: setting
  /// the list resets it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_categories_end(mut self, val: ListEnd) -> Self {
    self.categories_end = val;
    self
  }

  /// In-place setter for `categories_end`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_categories_end(&mut self, val: ListEnd) -> &mut Self {
    self.categories_end = val;
    self
  }
}

/// How an [`ImageAnalysis`]'s description ends.
///
/// A task that caps the description's length — the canonical
/// `ImageAnalysisTask` states its cap as the schema's `maxLength` — has
/// it closed by a constrained decoder at the cap, wherever the sentence
/// had got to (findit-studio/application#235:
/// `…facing forward with a neutral,略`). Only the decoder knows whether it
/// closed a string or the model did: an answer that ends exactly at the
/// cap reads the same either way. So this fact comes from the engine's
/// account ([`FieldEnd`], handed to `ImageAnalysisTask` through
/// [`Task::parse_ended`](crate::Task::parse_ended) or
/// `ImageAnalysisTask::parse_with_description_end`), and without one it
/// is [`Unknown`](Self::Unknown) and the text is kept as written.
///
/// The parser never cuts a description back: whether a ragged one still
/// holds whole sentences is a reading of its text, a consumer's choice to
/// make over the marked text, never the parser's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum DescriptionEnd {
  /// Nothing says how the description ended: it was parsed without the
  /// decoder's account (`Task::parse`), or built by hand. The default; the
  /// text is as the answer wrote it.
  #[default]
  Unknown,
  /// The model ended the description itself, whatever its length.
  Whole,
  /// The grammar closed the description at the cap: it ends where the cap
  /// stopped it, kept as written — less only the bytes of a last token the
  /// decoder names as cut ([`FieldEnd::with_cut`]).
  Ragged,
}

/// How one of an [`ImageAnalysis`]'s extension lists ends: `subjects`,
/// `objects`, `actions`, `emotion`, `lighting` or `categories`.
///
/// A task that caps a list's length — the canonical `ImageAnalysisTask`
/// states each list extension's cap as the schema's `maxItems` — has a
/// constrained decoder close the list once it holds that many items, however
/// many more the model would have listed (findit-studio/llmtask#18: a small
/// model kept appending `categories` until the token budget ran out). A
/// list's end, unlike the description's ([`DescriptionEnd`]), is read from
/// the answer itself: its items are counted against the cap, and an answer
/// cut off inside the list shows where it stops.
///
/// What the answer cannot show is whether the model would have gone on once
/// the list reached its cap: such a list is [`Capped`](Self::Capped) whether
/// the grammar closed it there or the model ended it at exactly that count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum ListEnd {
  /// Nothing says how the list ended: the task did not ask for it, or it was
  /// built by hand, or read from a document that does not record its end.
  /// The default.
  #[default]
  Unknown,
  /// The answer closed the list holding fewer items than its cap: no item
  /// was dropped for the list's length.
  Whole,
  /// The list reached its cap, or the answer was cut off inside it. It holds
  /// the items the answer listed first — no more than the cap, and only
  /// those written whole before a cut — and the model may have had more.
  Capped,
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

  /// LAW (Codex R2, [medium]): **setting or clearing the description
  /// resets how it ends.** The mark describes the text it was settled
  /// with; a new text — or none — is `Unknown` until a caller with the
  /// decoder's account sets it again.
  #[test]
  fn the_description_mutators_reset_how_it_ends() {
    let ragged = ImageAnalysis::new()
      .with_description("A cat sleeps on a")
      .with_description_end(DescriptionEnd::Ragged);
    assert_eq!(ragged.description_end(), DescriptionEnd::Ragged);
    assert_eq!(
      ragged
        .clone()
        .with_description("A cat sleeps on a rug.")
        .description_end(),
      DescriptionEnd::Unknown
    );
    let mut set = ragged.clone();
    set.set_description("A cat sleeps.");
    assert_eq!(set.description_end(), DescriptionEnd::Unknown);
    let mut cleared = ragged;
    cleared.set_description("");
    assert_eq!(cleared.description_end(), DescriptionEnd::Unknown);
    assert_eq!(
      cleared
        .with_description("A cat.")
        .with_description_end(DescriptionEnd::Whole)
        .description_end(),
      DescriptionEnd::Whole,
      "set after the text, the mark stands"
    );
  }

  /// LAW (Codex R4, [high]): **a document 0.4.1 wrote reads, its end
  /// unknown.** The JSON below is the shape 0.4.1 serializes, its ten keys
  /// in their order and no `description_end`: it reads every field as
  /// written and the mark as [`DescriptionEnd::Unknown`].
  #[cfg(all(feature = "serde", feature = "json"))]
  #[test]
  fn a_document_0_4_1_wrote_reads_with_its_end_unknown() {
    let written_by_0_4_1 = r#"{"scene":"kitchen","description":"A cat sleeps on a","subjects":["cat"],"objects":["rug"],"actions":["sleeping"],"emotion":["calm"],"shot_type":"wide","lighting":["soft"],"tags":["cat","rug"],"categories":["animal"]}"#;
    let read: ImageAnalysis =
      serde_json::from_str(written_by_0_4_1).expect("a 0.4.1 document reads");
    assert_eq!(read.description_end(), DescriptionEnd::Unknown);
    assert_eq!(read.description(), "A cat sleeps on a");
    assert_eq!(read.scene(), "kitchen");
    assert_eq!(read.subjects(), ["cat"]);
    assert_eq!(read.shot_type(), "wide");
    assert_eq!(read.categories(), ["animal"]);
  }

  /// LAW (Codex R4, [high]): **each mark is written after every field
  /// written before it existed, and round-trips by its snake-case name.** A
  /// positional format reads fields by place, so a mark between two older
  /// fields would present an older payload's next field where the mark is
  /// read; the order is asserted on the JSON written, key for key: the ten
  /// fields 0.4.x wrote, `description_end`, then the six list ends.
  #[cfg(all(feature = "serde", feature = "json"))]
  #[test]
  fn the_marks_are_written_after_the_fields_before_them_and_round_trip() {
    let marked = ImageAnalysis::new()
      .with_scene("kitchen")
      .with_description("A cat sleeps on a")
      .with_description_end(DescriptionEnd::Ragged)
      .with_subjects(vec!["cat".into()])
      .with_subjects_end(ListEnd::Whole)
      .with_tags(vec!["cat".into(), "rug".into()])
      .with_categories(vec!["animal".into(), "pet".into(), "home".into()])
      .with_categories_end(ListEnd::Capped);
    let written = serde_json::to_string(&marked).expect("an analysis serializes");
    assert_eq!(
      written,
      r#"{"scene":"kitchen","description":"A cat sleeps on a","subjects":["cat"],"objects":[],"actions":[],"emotion":[],"shot_type":"","lighting":[],"tags":["cat","rug"],"categories":["animal","pet","home"],"description_end":"ragged","subjects_end":"whole","objects_end":"unknown","actions_end":"unknown","emotion_end":"unknown","lighting_end":"unknown","categories_end":"capped"}"#
    );
    let read: ImageAnalysis = serde_json::from_str(&written).expect("it reads back");
    assert_eq!(read, marked);
  }

  /// LAW: **a document 0.5.2 wrote reads, its list ends unknown.** The JSON
  /// below is the shape 0.5.2 serializes, its eleven keys in their order and
  /// no list end: it reads every field as written and every list end as
  /// [`ListEnd::Unknown`].
  #[cfg(all(feature = "serde", feature = "json"))]
  #[test]
  fn a_document_0_5_2_wrote_reads_with_its_list_ends_unknown() {
    let written_by_0_5_2 = r#"{"scene":"kitchen","description":"A cat sleeps on a","subjects":["cat"],"objects":["rug"],"actions":["sleeping"],"emotion":["calm"],"shot_type":"wide","lighting":["soft"],"tags":["cat","rug"],"categories":["animal"],"description_end":"ragged"}"#;
    let read: ImageAnalysis =
      serde_json::from_str(written_by_0_5_2).expect("a 0.5.2 document reads");
    assert_eq!(read.description_end(), DescriptionEnd::Ragged);
    assert_eq!(read.subjects(), ["cat"]);
    assert_eq!(read.categories(), ["animal"]);
    for (name, end, ..) in lists() {
      assert_eq!(end(&read), ListEnd::Unknown, "{name}");
    }
  }

  /// A list's accessors: its name, how it ends, its builder and in-place
  /// setters, and its end's builder setter.
  type List = (
    &'static str,
    fn(&ImageAnalysis) -> ListEnd,
    fn(ImageAnalysis, Vec<SmolStr>) -> ImageAnalysis,
    fn(&mut ImageAnalysis, Vec<SmolStr>) -> &mut ImageAnalysis,
    fn(ImageAnalysis, ListEnd) -> ImageAnalysis,
  );

  /// The six lists whose end an [`ImageAnalysis`] records, in field order.
  fn lists() -> [List; 6] {
    [
      (
        "subjects",
        ImageAnalysis::subjects_end,
        ImageAnalysis::with_subjects,
        ImageAnalysis::set_subjects,
        ImageAnalysis::with_subjects_end,
      ),
      (
        "objects",
        ImageAnalysis::objects_end,
        ImageAnalysis::with_objects,
        ImageAnalysis::set_objects,
        ImageAnalysis::with_objects_end,
      ),
      (
        "actions",
        ImageAnalysis::actions_end,
        ImageAnalysis::with_actions,
        ImageAnalysis::set_actions,
        ImageAnalysis::with_actions_end,
      ),
      (
        "emotion",
        ImageAnalysis::emotion_end,
        ImageAnalysis::with_emotion,
        ImageAnalysis::set_emotion,
        ImageAnalysis::with_emotion_end,
      ),
      (
        "lighting",
        ImageAnalysis::lighting_end,
        ImageAnalysis::with_lighting,
        ImageAnalysis::set_lighting,
        ImageAnalysis::with_lighting_end,
      ),
      (
        "categories",
        ImageAnalysis::categories_end,
        ImageAnalysis::with_categories,
        ImageAnalysis::set_categories,
        ImageAnalysis::with_categories_end,
      ),
    ]
  }

  /// LAW: **setting or clearing a list resets how it ends**, as setting the
  /// description resets its mark: the mark describes the list it was
  /// recorded with, and a new list — or none — is `Unknown` until a caller
  /// sets it again. Each list's mark is its own, and the description's
  /// setters leave every list's.
  #[test]
  fn the_list_mutators_reset_how_the_list_ends() {
    for (name, end, with_list, set_list, with_end) in lists() {
      assert_eq!(end(&ImageAnalysis::new()), ListEnd::Unknown, "{name}");
      let marked = with_end(
        with_list(ImageAnalysis::new(), vec!["a".into()]),
        ListEnd::Capped,
      );
      assert_eq!(end(&marked), ListEnd::Capped, "{name}");
      assert_eq!(
        end(&with_list(marked.clone(), vec!["b".into()])),
        ListEnd::Unknown,
        "{name}: a new list"
      );
      let mut cleared = marked.clone();
      set_list(&mut cleared, Vec::new());
      assert_eq!(end(&cleared), ListEnd::Unknown, "{name}: cleared");
      for (other, _, other_with_list, ..) in lists() {
        if other != name {
          assert_eq!(
            end(&other_with_list(marked.clone(), vec!["c".into()])),
            ListEnd::Capped,
            "{name} beside {other}"
          );
        }
      }
      assert_eq!(
        end(&marked.clone().with_description("A cat.")),
        ListEnd::Capped,
        "{name} beside the description"
      );
      assert_eq!(
        end(&with_end(
          with_list(marked, vec!["d".into()]),
          ListEnd::Whole
        )),
        ListEnd::Whole,
        "{name}: set after the list, the mark stands"
      );
    }
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
pub use image_analysis_task::{Extension, ImageAnalysisTask, UnknownExtension};

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
    str::FromStr,
  };

  use serde::de::{self, Deserializer as _, MapAccess, Visitor};
  use serde_json::{Map, Value, json, value::RawValue};
  use smol_str::SmolStr;
  // Heap types under both std (resolves via the `extern crate std`) and
  // alloc-only (resolves via the `extern crate alloc as std` alias in
  // lib.rs).
  use std::{collections::BTreeMap, string::String, vec::Vec};

  use super::{DescriptionEnd, ImageAnalysis, ListEnd};
  use crate::{
    grammar::Grammar,
    task::{FieldCaps, FieldEnd, FieldEnds, JsonParseError, Task},
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
  ///
  /// # Names
  ///
  /// An extension's name is its field's JSON key, so a document can list
  /// the extensions a deployment switches on (`["scene", "shot_type"]`).
  /// Every road spells the same name: [`as_str`](Self::as_str) and
  /// [`Display`](core::fmt::Display) write it, [`FromStr`](core::str::FromStr)
  /// and [`TryFrom<&str>`](core::convert::TryFrom) read it back, and with the
  /// `serde` feature [`Serialize`](serde::Serialize) and
  /// [`Deserialize`](serde::Deserialize) carry it as a string. A name is read
  /// exactly as written: another case, surrounding whitespace or `-` for `_`
  /// is refused as [`UnknownExtension`], which names it, and so are
  /// `description` and `tags`, which are not extensions.
  ///
  /// | Extension | Name | JSON type | Default cap |
  /// | --- | --- | --- | --- |
  /// | [`Scene`](Self::Scene) | `scene` | string | |
  /// | [`Subjects`](Self::Subjects) | `subjects` | array of strings | 8 items |
  /// | [`Objects`](Self::Objects) | `objects` | array of strings | 8 items |
  /// | [`Actions`](Self::Actions) | `actions` | array of strings | 8 items |
  /// | [`Emotion`](Self::Emotion) | `emotion` | array of strings | 3 items |
  /// | [`ShotType`](Self::ShotType) | `shot_type` | string | |
  /// | [`Lighting`](Self::Lighting) | `lighting` | array of strings | 3 items |
  /// | [`Categories`](Self::Categories) | `categories` | array of strings | 3 items |
  ///
  /// [`Extension::ALL`] lists the extensions and [`Extension::NAMES`] their
  /// names, both in this order. Each list extension is capped at
  /// [`ImageAnalysisTask::max_items`] labels, by default
  /// [`ImageAnalysisTask::default_max_items`].
  ///
  /// ```
  /// use llmtask::image_analysis::{Extension, ImageAnalysisTask};
  ///
  /// // The names a deployment's document lists.
  /// let extensions = ["scene", "shot_type"]
  ///   .into_iter()
  ///   .map(str::parse)
  ///   .collect::<Result<Vec<Extension>, _>>()
  ///   .expect("both are extension names");
  /// assert_eq!(extensions, [Extension::Scene, Extension::ShotType]);
  /// assert_eq!(Extension::ShotType.to_string(), "shot_type");
  ///
  /// let task = ImageAnalysisTask::new().with_extensions(extensions);
  /// assert!(task.has_extension(Extension::ShotType));
  ///
  /// // Any other name is refused by name.
  /// let err = "shot-type".parse::<Extension>().unwrap_err();
  /// assert_eq!(err.name(), "shot-type");
  /// ```
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

  // `ImageAnalysisTask` keeps each extension's cap at `extension as usize`.
  const _: () = {
    let mut index = 0;
    while index < Extension::ALL.len() {
      assert!(Extension::ALL[index] as usize == index);
      index += 1;
    }
  };

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

    /// Every extension's name, in [`Extension::ALL`] order: `NAMES[i]` is
    /// `ALL[i].as_str()`. These eight are the names
    /// [`FromStr`](core::str::FromStr) reads; [`UnknownExtension`] lists
    /// them.
    pub const NAMES: [&'static str; 8] = {
      let mut names = [""; 8];
      let mut index = 0;
      while index < names.len() {
        names[index] = Self::ALL[index].as_str();
        index += 1;
      }
      names
    };

    /// The extension's name: its field's JSON key, which is also the name
    /// of the field's [`ImageAnalysis`] accessor. [`Display`](core::fmt::Display)
    /// writes it and [`FromStr`](core::str::FromStr) reads it back.
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

  impl fmt::Display for Extension {
    /// Writes the extension's name, [`Extension::as_str`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str(self.as_str())
    }
  }

  impl FromStr for Extension {
    type Err = UnknownExtension;

    /// Reads the extension whose [`as_str`](Extension::as_str) is `name`
    /// exactly, one of [`Extension::NAMES`]; any other `name` is refused as
    /// [`UnknownExtension`].
    fn from_str(name: &str) -> Result<Self, UnknownExtension> {
      Self::ALL
        .into_iter()
        .find(|extension| extension.as_str() == name)
        .ok_or_else(|| UnknownExtension {
          name: SmolStr::new(name),
        })
    }
  }

  impl TryFrom<&str> for Extension {
    type Error = UnknownExtension;

    /// The same as [`FromStr`](core::str::FromStr).
    fn try_from(name: &str) -> Result<Self, UnknownExtension> {
      name.parse()
    }
  }

  /// Serializes the extension as its name, a string.
  #[cfg(feature = "serde")]
  #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
  impl serde::Serialize for Extension {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
      S: serde::Serializer,
    {
      serializer.serialize_str(self.as_str())
    }
  }

  /// Deserializes an extension from its name, a string read as
  /// [`FromStr`](core::str::FromStr) reads it. Any other string is refused
  /// as an unknown variant, with the eight names expected, and a value that
  /// is not a string as an invalid type.
  #[cfg(feature = "serde")]
  #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
  impl<'de> serde::Deserialize<'de> for Extension {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
      D: serde::Deserializer<'de>,
    {
      deserializer.deserialize_str(NameVisitor)
    }
  }

  /// [`Visitor`] behind [`Extension`]'s `Deserialize`: one extension name.
  #[cfg(feature = "serde")]
  struct NameVisitor;

  #[cfg(feature = "serde")]
  impl Visitor<'_> for NameVisitor {
    type Value = Extension;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("an image-analysis extension name")
    }

    fn visit_str<E>(self, name: &str) -> Result<Extension, E>
    where
      E: de::Error,
    {
      name
        .parse()
        .map_err(|_| E::unknown_variant(name, &Extension::NAMES))
    }
  }

  /// The error [`Extension`]'s [`FromStr`](core::str::FromStr) and
  /// [`TryFrom<&str>`](core::convert::TryFrom) return for a name that is not
  /// one of [`Extension::NAMES`]. It carries the name as given, and its
  /// message names it and lists the eight.
  #[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
  #[error("unknown image-analysis extension {name:?}: expected one of {names}", names = NameList)]
  pub struct UnknownExtension {
    name: SmolStr,
  }

  impl UnknownExtension {
    /// The refused name, exactly as it was given.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn name(&self) -> &str {
      &self.name
    }
  }

  /// [`Extension::NAMES`] as [`UnknownExtension`]'s message lists them:
  /// comma-separated, in order.
  struct NameList;

  impl fmt::Display for NameList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      for (index, name) in Extension::NAMES.into_iter().enumerate() {
        if index > 0 {
          f.write_str(", ")?;
        }
        f.write_str(name)?;
      }
      Ok(())
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
  /// rewrites the model's words to make them fit: a description or a `tags`
  /// list over its cap is refused, and a list extension over its cap loses
  /// whole labels past it, never part of one.
  ///
  /// # The list extensions
  ///
  /// Each list extension — `subjects`, `objects`, `actions`, `emotion`,
  /// `lighting` and `categories` — holds at most
  /// [`max_items`](Self::max_items) labels, stated in the schema as its
  /// `maxItems` (the prompt does not state it). A constrained decoder closes
  /// the list at its cap however long the model would have gone on, so a
  /// model that keeps appending labels cannot run the answer out of tokens
  /// inside one (findit-studio/llmtask#18). The caps default by what each
  /// list holds ([`default_max_items`](Self::default_max_items)); a
  /// deployment sets its own with [`with_max_items`](Self::with_max_items).
  ///
  /// `parse` reads such a list to its cap rather than refusing it. The list
  /// is read whole first, so an element that is not a string refuses the
  /// field by name; then it keeps its first `max_items` elements as the
  /// answer wrote them — the cap counts elements as `maxItems` does, before
  /// labels are trimmed and deduplicated — and the [`ImageAnalysis`] records
  /// how it ended as its [`ListEnd`]: `Capped` when the answer listed at
  /// least the cap's count, `Whole` when it closed the list short of it.
  ///
  /// # An answer cut off inside a list
  ///
  /// A token budget can end an answer before its JSON closes. Such a text is
  /// not JSON and is refused as [`JsonParseError::Json`], with one exception:
  /// a text that is JSON as far as it goes and ends inside the array of a
  /// list extension the task asks for — after its `[`, after an item, after
  /// a comma, or inside a string item — is read as the answer closed after
  /// that list's last whole item. A string item the cut left open is
  /// dropped, the array and the object are closed, the closed answer goes
  /// through every check any answer goes through, and the cut list is
  /// `Capped`.
  ///
  /// - **Read:** the cut list is the last field the task asks for in the
  ///   answer, so every other field was written whole before it — as an
  ///   engine that writes the members in the schema's order writes
  ///   `categories`, the last field in [`ImageAnalysis`] order.
  /// - **Refused by name:** a field the task asks for that the cut left
  ///   unwritten is [`JsonParseError::MissingFields`] naming it, and the
  ///   closed answer's other refusals stand: an undeclared key, a duplicated
  ///   key, a value of another type.
  /// - **Not JSON:** a cut anywhere else — before the object opens, in a key
  ///   or before its value, inside or after a string field, inside or after
  ///   `tags`, after a list's own `]`, inside an item that is not a string,
  ///   inside a list the task does not ask for — is [`JsonParseError::Json`].
  ///
  /// # Example
  ///
  /// ```
  /// use core::num::NonZeroUsize;
  ///
  /// use llmtask::{
  ///   JsonParseError, ListEnd, Task,
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
  ///
  /// // A list extension is read to its cap: the labels past it are dropped
  /// // whole, and the analysis records that the list was capped.
  /// let task = ImageAnalysisTask::new()
  ///   .with_extensions([Extension::Categories])
  ///   .with_max_items(Extension::Categories, NonZeroUsize::new(2).unwrap());
  /// let raw = r#"{
  ///   "description": "Two people talk across a desk.", "tags": ["office"],
  ///   "categories": ["work", "business", "meeting"]
  /// }"#;
  /// let analysis = task.parse(raw).expect("a list over its cap is read to it");
  /// assert_eq!(analysis.categories(), ["work", "business"]);
  /// assert_eq!(analysis.categories_end(), ListEnd::Capped);
  ///
  /// // An answer cut off inside its last list keeps the items written whole.
  /// let raw = r#"{"description": "Two people talk.", "tags": ["office"], "categories": ["work", "busi"#;
  /// let analysis = task.parse(raw).expect("the cut list is the answer's last field");
  /// assert_eq!(analysis.categories(), ["work"]);
  /// assert_eq!(analysis.categories_end(), ListEnd::Capped);
  /// ```
  #[derive(Clone)]
  pub struct ImageAnalysisTask {
    // One `Extension::bit` per switched-on extension.
    extensions: u8,
    description_max_chars: NonZeroUsize,
    tags_max_items: NonZeroUsize,
    // Each extension's `max_items`, at `extension as usize`: `None` for the
    // two string extensions.
    list_max_items: [Option<NonZeroUsize>; Extension::ALL.len()],
    accept_empty: bool,
    // Both derived from `extensions` and the caps by `rebuild`, which every
    // setter of those calls, so the schema, the prompt and `parse` always
    // describe the same roster.
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

    /// [`Self::default_max_items`] of the lists that name one quality of the
    /// whole picture.
    const QUALITY_MAX_ITEMS: NonZeroUsize = NonZeroUsize::new(3).unwrap();

    /// The [`max_items`](Self::max_items) a task gives `extension`'s list
    /// unless [`with_max_items`](Self::with_max_items) sets another, or
    /// `None` for `scene` and `shot_type`, which are single strings.
    ///
    /// Each default is sized from what the list holds. `subjects`, `objects`
    /// and `actions` inventory what the picture shows, as `tags` does, and
    /// take [`Self::DEFAULT_TAGS_MAX_ITEMS`]; `emotion`, `lighting` and
    /// `categories` each name one quality of the whole picture — its tone,
    /// its light, its broad kind — in a handful of words, and take 3.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn default_max_items(extension: Extension) -> Option<NonZeroUsize> {
      match extension {
        Extension::Scene | Extension::ShotType => None,
        Extension::Subjects | Extension::Objects | Extension::Actions => {
          Some(Self::DEFAULT_TAGS_MAX_ITEMS)
        }
        Extension::Emotion | Extension::Lighting | Extension::Categories => {
          Some(Self::QUALITY_MAX_ITEMS)
        }
      }
    }

    /// Construct the default task: `description` and `tags` only, capped
    /// at [`Self::DEFAULT_DESCRIPTION_MAX_CHARS`] characters and
    /// [`Self::DEFAULT_TAGS_MAX_ITEMS`] tags, with no extension switched
    /// on (each list extension capped at its [`Self::default_max_items`]
    /// once it is) and `accept_empty = false` (a payload that lacks the
    /// required indexable content is treated as a model regression and
    /// rejected; see [`Self::with_accept_empty`] for the full predicate and
    /// the opt-in alternative).
    pub fn new() -> Self {
      let mut task = Self {
        extensions: 0,
        description_max_chars: Self::DEFAULT_DESCRIPTION_MAX_CHARS,
        tags_max_items: Self::DEFAULT_TAGS_MAX_ITEMS,
        list_max_items: Extension::ALL.map(Self::default_max_items),
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
    /// description by name, and settles the end of one that reached the cap
    /// — where a constrained decoder closed it — recording what it did as
    /// the analysis's [`DescriptionEnd`](crate::DescriptionEnd).
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

    /// The most labels `extension`'s list may hold, or `None` for `scene`
    /// and `shot_type`, which are single strings. Stated in the schema as
    /// the list's `maxItems`, so a constrained decoder closes the list there;
    /// `parse` reads a longer list to it and records how the list ended as
    /// its [`ListEnd`]. The prompt does not state it.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn max_items(&self, extension: Extension) -> Option<NonZeroUsize> {
      self.list_max_items[extension as usize]
    }

    /// Builder-style setter for `extension`'s [`max_items`](Self::max_items),
    /// to raise a cap a deployment finds too low, or to lower one. The schema
    /// and the prompt are rebuilt to match. `scene` and `shot_type` hold no
    /// items: for them this changes nothing, and their `max_items` stays
    /// `None`.
    pub fn with_max_items(mut self, extension: Extension, val: NonZeroUsize) -> Self {
      self.set_max_items(extension, val);
      self
    }

    /// In-place setter for `extension`'s [`max_items`](Self::max_items). See
    /// [`Self::with_max_items`].
    pub fn set_max_items(&mut self, extension: Extension, val: NonZeroUsize) -> &mut Self {
      let cap = &mut self.list_max_items[extension as usize];
      if cap.is_some() {
        *cap = Some(val);
        self.rebuild();
      }
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

    // --- parsing with the decoder's account ---

    /// [`Task::parse`], with the constrained decoder's account of how it
    /// ended the description: whether the grammar closed it at the
    /// field's `maxLength`, and the text of a last token it names as cut.
    ///
    /// `Task::parse` has the answer's text alone, and text cannot tell a
    /// string the grammar closed at the cap from one the model ended at
    /// exactly that length — so it keeps the description as written and
    /// marks it [`DescriptionEnd::Unknown`]. With `end`, an account of one
    /// string ([`FieldEnd::source`]):
    ///
    /// - a description that is not exactly that string is not one the
    ///   account describes: kept as written, [`DescriptionEnd::Unknown`];
    /// - the model ended it: kept as written, [`DescriptionEnd::Whole`],
    ///   whatever its length;
    /// - the grammar closed it at the cap: it holds exactly the task's
    ///   `description_max_chars` in characters, or the account was taken
    ///   under another cap and is refused by name; kept as written,
    ///   [`DescriptionEnd::Ragged`] — less exactly the bytes
    ///   [`FieldEnd::cut`] names, the text of a last token the decoder
    ///   reports cut, when the description ends with those bytes and leaves
    ///   text before them.
    ///
    /// That suffix is the one change a parse may make to a description's
    /// text. Nothing is removed for its script, its punctuation or its
    /// length, and nothing is cut back to a sentence end: whether a ragged
    /// description holds whole sentences is a consumer's reading of the
    /// marked text.
    ///
    /// Settling is idempotent on the TEXT: a settled description parsed
    /// again with the same account settles to the same text. Its mark stays
    /// where the settled text is still the account's string — nothing was
    /// removed or trimmed — and is [`DescriptionEnd::Unknown`] where it is
    /// not: the account is not about the settled string.
    ///
    /// # Errors
    ///
    /// As [`Task::parse`]; and [`JsonParseError::DescriptionCapMismatch`]
    /// when `end` is a cap's account of the description whose characters
    /// are not this task's cap.
    pub fn parse_with_description_end(
      &self,
      raw: &str,
      end: &FieldEnd,
    ) -> Result<ImageAnalysis, JsonParseError> {
      self.parse_answer(raw, Some(end))
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
    /// enforces this promise itself via `unknown_fields`, each field's type
    /// and the caps it refuses a value over via `usable_value`, and each list
    /// extension's cap via `settle_list`, which reads the list to it.
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
    /// [`Shape`], plus the task's cap on it — `maxLength` on `description`,
    /// `maxItems` on `tags` and on each list extension.
    ///
    /// No other keyword: an engine refuses a schema that uses one it does
    /// not implement, and llguidance implements no `uniqueItems`, so the
    /// labels are deduplicated by `parse` rather than by the grammar.
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
        | Field::Categories => {
          if let Some(cap) = self.list_cap(field) {
            property["maxItems"] = cap.get().into();
          }
        }
      }
      property
    }

    /// The cap on `field`'s labels when `field` is a list extension's, its
    /// [`max_items`](Self::max_items); `None` for every other field —
    /// `tags` has its own cap, which `parse` refuses a longer list over.
    fn list_cap(&self, field: Field) -> Option<NonZeroUsize> {
      Extension::ALL
        .into_iter()
        .find(|extension| extension.field() == field)
        .and_then(|extension| self.max_items(extension))
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
    /// field (see [`decode_field`]), and within any cap the task refuses a
    /// longer value over (see [`Self::exceeds_cap`]). `Ok(None)` leaves the
    /// field unusable: absent, `null`, of any other JSON type, or over such
    /// a cap. Keys outside the roster are a separate concern, handled by
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
    /// model's words to fit. A list extension's cap is not such a cap:
    /// [`Self::settle_list`] reads the list to it, dropping whole labels.
    /// `parse` names every unusable field at once.
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
    /// `parse` trims it), `tags` in elements of the array. A list
    /// extension's cap never makes its value unusable: [`Self::settle_list`]
    /// reads the list to it.
    fn exceeds_cap(&self, field: Field, value: &FieldValue) -> bool {
      match (field, value) {
        (Field::Description, FieldValue::String(description)) => {
          description.chars().count() > self.description_max_chars.get()
        }
        (Field::Tags, FieldValue::Strings(tags)) => tags.len() > self.tags_max_items.get(),
        _ => false,
      }
    }

    /// `written`, the description as the answer wrote it, as
    /// [`ImageAnalysis`] holds it, and how it ends ([`DescriptionEnd`]),
    /// settled by the decoder's account `ended` — see
    /// [`Self::parse_with_description_end`]. The text is kept as written,
    /// trimmed as every parse trims it, except for the one suffix a cap's
    /// account names by its bytes: nothing is inferred from the text's
    /// length, its script or its punctuation.
    fn settle_description(
      &self,
      written: &str,
      ended: Option<&FieldEnd>,
    ) -> Result<(SmolStr, DescriptionEnd), JsonParseError> {
      // An account settles the one string it holds, byte for byte; about any
      // other text it says nothing.
      let Some(ended) = ended.filter(|ended| written == ended.source()) else {
        return Ok((SmolStr::new(written.trim()), DescriptionEnd::Unknown));
      };
      if !ended.closed_at_cap() {
        return Ok((SmolStr::new(written.trim()), DescriptionEnd::Whole));
      }
      // The grammar closes a string exactly at the cap: a cap's account of
      // a string of another length was taken under another cap.
      let (cap, chars) = (self.description_max_chars.get(), written.chars().count());
      if chars != cap {
        return Err(JsonParseError::DescriptionCapMismatch { cap, chars });
      }
      let kept = ended
        .cut()
        .filter(|suffix| !suffix.is_empty())
        .and_then(|suffix| written.strip_suffix(suffix))
        .map(str::trim)
        .filter(|kept| !kept.is_empty())
        .unwrap_or_else(|| written.trim());
      Ok((SmolStr::new(kept), DescriptionEnd::Ragged))
    }

    /// A list extension's `value` as [`ImageAnalysis`] holds it, and how the
    /// list ends ([`ListEnd`]). The cap counts the elements as the answer
    /// wrote them, as the schema's `maxItems` does: a list of at least the
    /// cap's count keeps its first `cap` elements and is `Capped`, a shorter
    /// one is kept whole and is `Whole`. A list the answer was cut off
    /// inside (`cut`) is `Capped` whatever its count. Only then are the
    /// kept elements made labels (see [`FieldValue::into_labels`]).
    fn settle_list(
      &self,
      field: Field,
      mut value: FieldValue,
      cut: Option<Field>,
    ) -> (Vec<SmolStr>, ListEnd) {
      let mut end = if cut == Some(field) {
        ListEnd::Capped
      } else {
        ListEnd::Whole
      };
      if let (FieldValue::Strings(strings), Some(cap)) = (&mut value, self.list_cap(field))
        && strings.len() >= cap.get()
      {
        strings.truncate(cap.get());
        end = ListEnd::Capped;
      }
      (value.into_labels(), end)
    }

    /// `raw` closed where a cut left one of its lists open, and that list's
    /// field: `raw` up to the end of the last item the list holds whole, then
    /// `]}`. `None` unless `raw` ends inside the array of a list extension
    /// the task asks for (see [`cut_list`]).
    fn close_cut_list(&self, raw: &str) -> Option<(String, Field)> {
      let (key, whole) = cut_list(raw)?;
      let field = self
        .fields()
        .find(|&field| field.key() == key && self.list_cap(field).is_some())?;
      let mut closed = String::from(raw.get(..whole)?);
      closed.push_str("]}");
      Some((closed, field))
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
      self.parse_answer(raw, None)
    }

    /// [`Task::parse`], with the description settled by its account in
    /// `ends` — the `"description"` entry — exactly as
    /// [`ImageAnalysisTask::parse_with_description_end`] settles it; with
    /// no such entry, as [`Task::parse`]. An account of any other field is
    /// ignored: the description is the one string the task caps. Every list
    /// is read from the answer's text as `Task::parse` reads it: its
    /// [`ListEnd`] needs no account.
    ///
    /// # Errors
    ///
    /// As [`ImageAnalysisTask::parse_with_description_end`] when `ends`
    /// holds a description account, else as [`Task::parse`].
    fn parse_ended(&self, raw: &str, ends: &FieldEnds) -> Result<Self::Output, JsonParseError> {
      self.parse_answer(raw, ends.get(Field::Description.key()))
    }

    /// The description's cap, [`ImageAnalysisTask::description_max_chars`]
    /// — the `maxLength` the schema states for it — so an engine reads the
    /// description's account against the cap the task settles it by. No
    /// other field is capped in characters: `tags` and the list extensions
    /// cap their items.
    fn field_caps(&self) -> FieldCaps {
      let mut caps = FieldCaps::new();
      caps.insert(Field::Description.key(), self.description_max_chars.get());
      caps
    }
  }

  impl ImageAnalysisTask {
    /// [`Task::parse`]'s body, with the decoder's account of how it ended
    /// the description when there is one.
    fn parse_answer(
      &self,
      raw: &str,
      description_end: Option<&FieldEnd>,
    ) -> Result<ImageAnalysis, JsonParseError> {
      // Not a plain `serde_json::from_str` into a `Value` (see
      // `parse_members`'s doc comment): that collapses a duplicate
      // top-level member — later overwrites earlier — before any check
      // below ever sees both copies, and it fails on valid JSON (a number
      // outside `f64`'s range, deep nesting) before the field holding it
      // can be named. `raw` goes in untrimmed: the deserializer skips the
      // JSON whitespace a JSON text may carry around its value, and any
      // other character there is not JSON.
      match parse_members(raw) {
        // JSON as far as it goes, but it ends too soon: an answer cut off
        // inside a list extension is read as the answer closed after that
        // list's last whole item; any other is not JSON.
        Err(JsonParseError::Json(err)) if err.is_eof() => match self.close_cut_list(raw) {
          Some((closed, cut)) => {
            self.read_answer(parse_members(&closed)?, Some(cut), description_end)
          }
          None => Err(JsonParseError::Json(err)),
        },
        members => self.read_answer(members?, None, description_end),
      }
    }

    /// The analysis the answer's top-level `members` hold (`None`: the
    /// answer is not a JSON object), with the decoder's account of how it
    /// ended the description when there is one; `cut` is the list extension
    /// the answer was cut off inside, if it was.
    fn read_answer(
      &self,
      members: Option<BTreeMap<String, &RawValue>>,
      cut: Option<Field>,
      description_end: Option<&FieldEnd>,
    ) -> Result<ImageAnalysis, JsonParseError> {
      let Some(members) = members else {
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
          Field::Description => {
            let (description, end) =
              self.settle_description(&value.into_string(), description_end)?;
            result.set_description(description).set_description_end(end)
          }
          Field::Subjects => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_subjects(labels).set_subjects_end(end)
          }
          Field::Objects => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_objects(labels).set_objects_end(end)
          }
          Field::Actions => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_actions(labels).set_actions_end(end)
          }
          Field::Emotion => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_emotion(labels).set_emotion_end(end)
          }
          Field::ShotType => result.set_shot_type(value.into_label()),
          Field::Lighting => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_lighting(labels).set_lighting_end(end)
          }
          Field::Tags => result.set_tags(value.into_labels()),
          Field::Categories => {
            let (labels, end) = self.settle_list(field, value, cut);
            result.set_categories(labels).set_categories_end(end)
          }
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

  /// When `raw` ends inside the array value of a member of its top-level
  /// object: that member's key, JSON-decoded, and how many bytes of `raw`
  /// run through the last item the array holds whole — through its `[`
  /// when it holds none. An item the text ends inside is dropped when it is
  /// a string; a text that ends inside any other item, or anywhere but in a
  /// member's array (before the object opens, in a key or before its value,
  /// inside or after another value), gives `None`.
  ///
  /// `raw` must be a text serde_json has read as JSON up to its end, which
  /// came too soon ([`serde_json::Error::is_eof`]): every byte up to there
  /// is where JSON grammar allows it, so this walk steps over values without
  /// checking them, and the text it keeps is read again whole, by
  /// [`parse_members`].
  fn cut_list(raw: &str) -> Option<(String, usize)> {
    let bytes = raw.as_bytes();
    let mut at = skip_whitespace(bytes, 0);
    if bytes.get(at) != Some(&b'{') {
      return None;
    }
    at = skip_whitespace(bytes, at + 1);
    loop {
      if bytes.get(at) != Some(&b'"') {
        return None;
      }
      let key_end = string_end(bytes, at)?;
      let key: String = serde_json::from_str(raw.get(at..key_end)?).ok()?;
      at = skip_whitespace(bytes, key_end);
      if bytes.get(at) != Some(&b':') {
        return None;
      }
      at = skip_whitespace(bytes, at + 1);
      if bytes.get(at) == Some(&b'[') {
        // Through the `[`: no item is whole yet.
        let mut whole = at + 1;
        at = skip_whitespace(bytes, whole);
        if bytes.get(at) == Some(&b']') {
          at += 1;
        } else {
          loop {
            let Some(&first) = bytes.get(at) else {
              return Some((key, whole));
            };
            let Some(end) = value_end(bytes, at) else {
              return (first == b'"').then_some((key, whole));
            };
            whole = end;
            at = skip_whitespace(bytes, end);
            match bytes.get(at) {
              None => return Some((key, whole)),
              Some(b',') => at = skip_whitespace(bytes, at + 1),
              Some(b']') => {
                at += 1;
                break;
              }
              Some(_) => return None,
            }
          }
        }
      } else {
        at = value_end(bytes, at)?;
      }
      at = skip_whitespace(bytes, at);
      if bytes.get(at) != Some(&b',') {
        return None;
      }
      at = skip_whitespace(bytes, at + 1);
    }
  }

  /// The first byte at or after `at` in `bytes` that is not JSON
  /// whitespace.
  fn skip_whitespace(bytes: &[u8], at: usize) -> usize {
    let mut index = at;
    while matches!(bytes.get(index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
      index += 1;
    }
    index
  }

  /// The byte just past the JSON string whose opening quote is at `at` in
  /// `bytes`, or `None` when the text ends inside it. A backslash opens a
  /// two-byte escape or `\u` and four hex digits, none of them a quote, so
  /// stepping over the byte after each backslash meets the closing quote.
  fn string_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut index = at + 1;
    loop {
      match *bytes.get(index)? {
        b'"' => return Some(index + 1),
        b'\\' => index += 2,
        _ => index += 1,
      }
    }
  }

  /// The byte just past the JSON value that starts at `at` in `bytes`, or
  /// `None` when no value starts there or the text ends inside it. A number
  /// or a literal that runs to the end of the text gives `None` too: the
  /// text's end cannot show that it is whole.
  fn value_end(bytes: &[u8], at: usize) -> Option<usize> {
    match *bytes.get(at)? {
      b'"' => string_end(bytes, at),
      b'[' | b'{' => {
        let mut depth = 0_usize;
        let mut index = at;
        loop {
          match *bytes.get(index)? {
            b'"' => index = string_end(bytes, index)?,
            b'[' | b'{' => {
              depth += 1;
              index += 1;
            }
            b']' | b'}' => {
              depth = depth.checked_sub(1)?;
              index += 1;
              if depth == 0 {
                return Some(index);
              }
            }
            _ => index += 1,
          }
        }
      }
      b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => bytes
        .get(at..)?
        .iter()
        .position(|byte| matches!(byte, b',' | b']' | b'}' | b' ' | b'\t' | b'\n' | b'\r'))
        .map(|len| at + len),
      _ => None,
    }
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
    /// A string field's value as the answer wrote it, untrimmed. `parse`
    /// asks this only of a string field's value, so the empty string for
    /// an array is never reached.
    fn into_string(self) -> String {
      match self {
        Self::String(string) => string,
        Self::Strings(_) => String::new(),
      }
    }

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

    // ===== the extension names =====

    /// LAW: every extension's name reads back as that extension, through
    /// `FromStr` and `TryFrom<&str>`; `Display` writes the name `as_str`
    /// returns; and `NAMES` lists the names in `ALL` order.
    #[test]
    fn every_extension_name_reads_back_as_that_extension() {
      for (index, extension) in Extension::ALL.into_iter().enumerate() {
        let name = extension.as_str();
        assert_eq!(Extension::NAMES[index], name);
        assert_eq!(
          name.parse::<Extension>(),
          Ok(extension),
          "{name} must read back through FromStr"
        );
        assert_eq!(
          Extension::try_from(name),
          Ok(extension),
          "{name} must read back through TryFrom<&str>"
        );
        assert_eq!(format!("{extension}"), name);
      }
    }

    /// LAW: any name but the eight is refused as `UnknownExtension`, which
    /// carries the name as given, and whose message names it and lists the
    /// eight. A name is read exactly as written, so another case,
    /// surrounding whitespace or `-` for `_` is unknown; and `description`
    /// and `tags`, which every task asks for, are not extensions.
    #[test]
    fn an_unknown_name_is_refused_by_name_listing_the_eight() {
      for name in [
        "",
        "Scene",
        "SCENE",
        " scene",
        "scene ",
        "shot-type",
        "shotType",
        "ShotType",
        "description",
        "tags",
        "mood",
        "scene,subjects",
      ] {
        let err = name
          .parse::<Extension>()
          .expect_err("only the eight names are extensions");
        assert_eq!(err.name(), name);
        assert_eq!(Extension::try_from(name), Err(err.clone()));
        assert_eq!(
          format!("{err}"),
          format!(
            "unknown image-analysis extension {name:?}: expected one of scene, subjects, objects, actions, emotion, shot_type, lighting, categories"
          )
        );
      }
    }

    /// LAW: with the `serde` feature an extension travels as its name. A
    /// list of extensions is a JSON array of names and reads back as the
    /// same list; a name `FromStr` refuses is an unknown variant, refused
    /// with the eight names expected; and a value that is not a string is
    /// refused.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_carries_an_extension_as_its_name() {
      let all: Vec<Extension> = Extension::ALL.into();
      let json = serde_json::to_string(&all).expect("a list of extensions serializes");
      assert_eq!(
        json,
        r#"["scene","subjects","objects","actions","emotion","shot_type","lighting","categories"]"#
      );
      let read: Vec<Extension> =
        serde_json::from_str(&json).expect("the array of names deserializes");
      assert_eq!(read, all);
      for extension in Extension::ALL {
        let name = extension.as_str();
        assert_eq!(
          serde_json::to_value(extension).expect("an extension serializes"),
          json!(name)
        );
        assert_eq!(
          serde_json::from_value::<Extension>(json!(name)).expect("its name deserializes"),
          extension
        );
      }

      let err = serde_json::from_str::<Vec<Extension>>(r#"["scene","shot-type"]"#)
        .expect_err("an unknown name is refused");
      assert!(
        format!("{err}").starts_with(
          "unknown variant `shot-type`, expected one of `scene`, `subjects`, `objects`, `actions`, `emotion`, `shot_type`, `lighting`, `categories`"
        ),
        "got {err}"
      );
      for refused in [
        r#""Scene""#,
        r#""description""#,
        r#""""#,
        "0",
        "true",
        "null",
        r#"["scene"]"#,
        r#"{"scene":null}"#,
      ] {
        assert!(
          serde_json::from_str::<Extension>(refused).is_err(),
          "{refused} is not an extension name"
        );
      }
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
    /// them. No other field carries a `maxLength`; every other array field —
    /// each list extension — carries its own `maxItems` (see
    /// `every_list_extension_states_its_cap_in_the_schema`), and `scene` and
    /// `shot_type` carry none.
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
        assert_eq!(
          property.get("maxItems").is_some(),
          property["type"] == "array",
          "{key}: every array field, and no other, carries maxItems"
        );
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
    /// and an answer at a cap parses whole: `parse` never truncates an
    /// answer to fit a cap, and without the decoder's account it does not
    /// know how the answer ended.
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
      assert_eq!(analysis.description_end(), crate::DescriptionEnd::Unknown);
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

    // ===== a description the decoder closed at the cap (application#235) =====

    /// The description `task` settles from `written`, and how it ends —
    /// through `Task::parse` when `end` is `None`, else through
    /// `parse_with_description_end`.
    fn settled(
      task: &ImageAnalysisTask,
      written: &str,
      end: Option<FieldEnd>,
    ) -> (String, crate::DescriptionEnd) {
      parsed(task, written, end).unwrap_or_else(|e| panic!("{written:?} must parse: {e:?}"))
    }

    /// [`settled`], with the parse's refusal kept.
    fn parsed(
      task: &ImageAnalysisTask,
      written: &str,
      end: Option<FieldEnd>,
    ) -> Result<(String, crate::DescriptionEnd), JsonParseError> {
      let answer = answer(written);
      let analysis = match end {
        None => task.parse(&answer),
        Some(end) => task.parse_with_description_end(&answer, &end),
      }?;
      Ok((analysis.description().into(), analysis.description_end()))
    }

    /// The default task's answer whose description is `written`.
    fn answer(written: &str) -> String {
      serde_json::to_string(&serde_json::json!({ "description": written, "tags": ["label"] }))
        .expect("a JSON value serializes")
    }

    /// The task whose cap `written` reaches: capped at its length.
    fn capped_at(written: &str) -> ImageAnalysisTask {
      ImageAnalysisTask::new().with_description_max_chars(nz(written.chars().count()))
    }

    /// The issue's two captions, the reviews' counter-examples, and the
    /// shapes the rules below turn on — each at its own cap.
    const AT_THE_CAP: [&str; 12] = [
      "A person wearing a white t-shirt with 'PLAYFUL OF CLASSIC' printed on it stands indoors, facing forward with a neutral,略",
      "A dimly lit room features a wooden chair with a white cushion, a small wooden desk with drawers, and a patterned rug on木",
      "A wooden sign is printed with 愛",
      "A portrait of Dr. Smith standing beside a desk in a quiet office",
      "A screen asks \"Ready?\" while runners wait at the line",
      "J. R. R. Tolkien's books stand on a shelf beside a lamp",
      "A cat sleeps on a rug\u{FFFD}",
      "Two people talk across a desk. A lamp glows on a shelf behind",
      "台所で二人が話している。窓の外は雨が降っていて、壁の時計は",
      "Two people talk across a desk.",
      "A portrait of Capt.",
      "A sign reads 'STOP! LOOK BOTH WAYS'",
    ];

    /// `written`'s last character, as the text an engine names when it
    /// reports that token cut.
    fn last_char(written: &str) -> &str {
      written
        .char_indices()
        .next_back()
        .map_or("", |(at, _)| &written[at..])
    }

    /// LAW (Codex R1): without the decoder's account nothing is deleted
    /// or inferred. Text cannot tell a string the grammar closed at the cap
    /// from one the model ended at exactly that length, a stray from a
    /// word in another script, or an abbreviation from a sentence end — so
    /// `Task::parse` keeps every description as written, `Unknown`.
    #[test]
    fn parse_without_the_decoders_account_deletes_nothing() {
      for written in AT_THE_CAP {
        assert_eq!(
          settled(&capped_at(written), written, None),
          (written.into(), crate::DescriptionEnd::Unknown),
          "{written}"
        );
      }
    }

    /// LAW: a description the model ended is `Whole` whatever its length —
    /// at the cap included — and kept as written; a cut boundary means
    /// nothing beside the model's own end.
    #[test]
    fn a_description_the_model_ended_is_whole_at_any_length() {
      for written in AT_THE_CAP {
        for end in [
          FieldEnd::model(written),
          FieldEnd::model(written).with_cut(last_char(written)),
        ] {
          assert_eq!(
            settled(&capped_at(written), written, Some(end.clone())),
            (written.into(), crate::DescriptionEnd::Whole),
            "{written} under {end:?}"
          );
        }
      }
    }

    /// LAW (Codex R2, the authority's order): a description the grammar
    /// closed at the cap is `Ragged` and kept as written — on a sentence
    /// end or not, CJK letter or not. Only the decoder can say whether a
    /// cap stopped it, and nothing in the text is taken for an ending.
    #[test]
    fn a_description_closed_at_the_cap_is_ragged_and_kept_as_written() {
      for written in AT_THE_CAP {
        assert_eq!(
          settled(&capped_at(written), written, Some(FieldEnd::cap(written))),
          (written.into(), crate::DescriptionEnd::Ragged),
          "{written}"
        );
      }
    }

    /// LAW (Codex R2, [high]): **the reviews' fixtures are kept and marked.**
    /// An abbreviation the allow-list did not hold (`Capt.`) and a
    /// single-quoted shout (`'STOP! LOOK BOTH WAYS'`) are not sentence ends
    /// the parser may cut at or call complete: closed at the cap they are
    /// kept as written and `Ragged`, the text continuing past them kept too.
    #[test]
    fn the_reviews_fixtures_are_kept_as_written_and_marked_ragged() {
      for written in [
        "A portrait of Capt.",
        "A portrait of Capt. Morgan standing nearby",
        "A sign reads 'STOP! LOOK BOTH WAYS' beside the road",
        "A sign reads 'STOP!",
      ] {
        assert_eq!(
          settled(&capped_at(written), written, Some(FieldEnd::cap(written))),
          (written.into(), crate::DescriptionEnd::Ragged),
          "{written}"
        );
      }
    }

    /// LAW (Codex R2–R3): **only the bytes the decoder names are removed**,
    /// and only where the description ends with them and keeps text before
    /// them. Every trailing U+FFFD is not a cut token: with no suffix named,
    /// nothing goes; a suffix the text does not end with — one whose last
    /// byte is the text's but whose character is not, the nearest a `&str`
    /// comes to a cut inside a character — an empty suffix, and a suffix
    /// that is the whole text all remove nothing.
    #[test]
    fn only_a_suffix_the_decoder_names_is_removed() {
      let written = "A cat sleeps on a rug\u{FFFD}\u{FFFD}";
      let task = capped_at(written);
      assert_eq!(
        settled(
          &task,
          written,
          Some(FieldEnd::cap(written).with_cut("\u{FFFD}"))
        ),
        (
          "A cat sleeps on a rug\u{FFFD}".into(),
          crate::DescriptionEnd::Ragged
        ),
        "exactly the named token's bytes go"
      );
      for (end, why) in [
        (FieldEnd::cap(written), "no suffix named"),
        (FieldEnd::cap(written).with_cut(""), "an empty suffix"),
        (
          FieldEnd::cap(written).with_cut("dog\u{FFFD}"),
          "a suffix the text does not end with",
        ),
        (FieldEnd::cap(written).with_cut(written), "the whole text"),
      ] {
        assert_eq!(
          settled(&task, written, Some(end)),
          (written.into(), crate::DescriptionEnd::Ragged),
          "{why}"
        );
      }
      // `é` is C3 A9 and `©` is C2 A9: the text's last byte is the
      // suffix's, its last character is not.
      let written = "A café";
      assert_eq!(
        settled(
          &capped_at(written),
          written,
          Some(FieldEnd::cap(written).with_cut("\u{A9}"))
        ),
        (written.into(), crate::DescriptionEnd::Ragged),
        "not a cut inside a character"
      );
    }

    /// LAW (Codex R5–R6, [medium]): **an account is about one string, byte
    /// for byte, and says nothing about any other.** A cap's account taken
    /// for `abc�` (the suffix `�`) removes nothing from `xyz�`, which has its
    /// length and ends with its suffix — a stale account after a retry, or
    /// one misassociated in a batch — nor from a description of another
    /// length or with a leading space, and marks each of them `Unknown`: the
    /// account does not describe how they end. A model's account of another
    /// string says nothing either. `abc�` itself, under the cap of four its
    /// account was taken under, still loses the `�`, `Ragged`.
    #[test]
    fn an_account_says_nothing_about_another_string() {
      let end = FieldEnd::cap("abc\u{FFFD}").with_cut("\u{FFFD}");
      assert_eq!(end.source(), "abc\u{FFFD}");
      assert_eq!(end.cut(), Some("\u{FFFD}"));
      for written in ["xyz\u{FFFD}", "abcd\u{FFFD}", " abc\u{FFFD}"] {
        for end in [end.clone(), FieldEnd::model("abc\u{FFFD}")] {
          assert_eq!(
            settled(&capped_at(written), written, Some(end.clone())),
            (written.trim().into(), crate::DescriptionEnd::Unknown),
            "{written:?} is not the string of {end:?}"
          );
        }
      }
      assert_eq!(
        settled(&capped_at("abc\u{FFFD}"), "abc\u{FFFD}", Some(end)),
        ("abc".into(), crate::DescriptionEnd::Ragged),
        "the account's own string loses the named bytes"
      );
    }

    /// LAW (Codex R6, [medium]): **a cap's account taken under another cap is
    /// refused by name.** The grammar closes a string exactly at the cap, so
    /// an account of the four-character `abc�` was taken under a cap of four.
    /// Reused on that very string by a task whose cap is 120 — the cap moved
    /// — it is refused as `DescriptionCapMismatch`, never settled to `abc`.
    /// The same account on ` abc�` is about another string, `Unknown`, and
    /// the settled `abc�` parsed again with it is refused by name in turn:
    /// the cap skew surfaces, and nothing is ever stripped.
    #[test]
    fn a_cap_account_taken_under_another_cap_is_refused_by_name() {
      let task = ImageAnalysisTask::new();
      assert_eq!(task.description_max_chars().get(), 120);
      let end = FieldEnd::cap("abc\u{FFFD}").with_cut("\u{FFFD}");
      let refused = |written: &str| {
        matches!(
          parsed(&task, written, Some(end.clone())),
          Err(JsonParseError::DescriptionCapMismatch { cap: 120, chars: 4 })
        )
      };
      assert!(refused("abc\u{FFFD}"), "the identical string under cap 120");
      let first = settled(&task, " abc\u{FFFD}", Some(end.clone()));
      assert_eq!(
        first,
        ("abc\u{FFFD}".into(), crate::DescriptionEnd::Unknown),
        "the string with a leading space is not the account's"
      );
      assert!(
        refused(&first.0),
        "parsed again, it is the account's string: refused"
      );
    }

    /// LAW (Codex R3, [high]; R6): **a trimmed description re-settles to
    /// the same text.** The account names the suffix by its bytes, so
    /// trimming moves nothing it means. The review's cap-7 `" abcdef"`,
    /// named whole past its leading space, keeps `"abcdef"` — nothing would
    /// be left before the suffix — and re-settles to `"abcdef"`, where an
    /// offset read again in trimmed coordinates cut it to `"a"`.
    /// `"xyz abcdef"` loses `"abcdef"` once and re-settles to `"xyz"`;
    /// leading whitespace before a cut CJK token settles to its letters and
    /// stays there; and a suffix repeated before itself (`"xaa"` less `"a"`)
    /// is removed once. Each settled text is not the string the account was
    /// taken from, so parsed again it is `Unknown`: the account is not about
    /// it.
    #[test]
    fn a_trimmed_description_resettles_to_itself() {
      for (written, suffix, once) in [
        (" abcdef", "abcdef", "abcdef"),
        ("xyz abcdef", "abcdef", "xyz"),
        ("  日本語\u{FFFD}", "\u{FFFD}", "日本語"),
        ("  日本語 ", "語 ", "日本"),
        // The named bytes repeat before themselves: the account describes the
        // string it was taken from, and a settled one is not it.
        ("A rug\u{FFFD}\u{FFFD}", "\u{FFFD}", "A rug\u{FFFD}"),
        ("xaa", "a", "xa"),
      ] {
        let task = capped_at(written);
        let end = FieldEnd::cap(written).with_cut(suffix);
        let first = settled(&task, written, Some(end.clone()));
        assert_eq!(
          first,
          (once.into(), crate::DescriptionEnd::Ragged),
          "{written:?}"
        );
        assert_eq!(
          settled(&task, &first.0, Some(end)),
          (once.into(), crate::DescriptionEnd::Unknown),
          "{written:?} re-settles to the same text, about which the account says nothing"
        );
      }
    }

    /// LAW (Codex R1–R6): settling is idempotent on the TEXT — a settled
    /// description parsed again with the same account settles to the same
    /// text, and parsed without one is kept as it stands — over the
    /// fixtures, and as a property over pseudo-random pairs of a text and a
    /// suffix drawn from spaces, ASCII, CJK and U+FFFD, where the settled
    /// text is always either the text trimmed or the text less exactly the
    /// named suffix, trimmed. The mark parsed again is the same where the
    /// settled text is still the account's string, and `Unknown` where it
    /// is not (R6). The account is bound to its text byte for byte (R5):
    /// the same text with one letter swapped for another of its byte length
    /// settles to itself trimmed under that account, `Unknown`.
    #[test]
    fn settling_is_idempotent() {
      let once_more = |written: &str, end: FieldEnd| {
        let task = capped_at(written);
        let once = settled(&task, written, Some(end.clone()));
        let twice = settled(&task, &once.0, Some(end.clone()));
        let mark = if once.0 == end.source() {
          once.1
        } else {
          crate::DescriptionEnd::Unknown
        };
        assert_eq!(twice, (once.0.clone(), mark), "{written:?} under {end:?}");
        assert_eq!(settled(&task, &once.0, None).0, once.0, "{written:?}");
        once
      };
      for written in AT_THE_CAP {
        for end in [
          FieldEnd::model(written),
          FieldEnd::cap(written),
          FieldEnd::cap(written).with_cut(last_char(written)),
        ] {
          once_more(written, end);
        }
      }

      const PIECES: [&str; 8] = [" ", "a", "b", "日", "本", "\u{FFFD}", "  ", "語"];
      let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
      let mut next = |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
      };
      for _ in 0..2_000 {
        // Some text the parse holds: a letter somewhere among the pieces.
        let mut pieces: Vec<&str> = (0..next(8)).map(|_| PIECES[next(PIECES.len())]).collect();
        let letter = ["a", "b", "日", "本", "語"][next(5)];
        let at = next(pieces.len() + 1);
        pieces.insert(at, letter);
        let text: String = pieces.concat();
        let suffix: String = (0..next(4)).map(|_| PIECES[next(PIECES.len())]).collect();
        let (settled, end) = once_more(&text, FieldEnd::cap(&text).with_cut(&suffix));
        assert_eq!(end, crate::DescriptionEnd::Ragged);
        let less = text
          .strip_suffix(suffix.as_str())
          .filter(|_| !suffix.is_empty())
          .map(str::trim)
          .filter(|kept| !kept.is_empty());
        assert!(
          settled == text.trim() || less == Some(settled.as_str()),
          "{text:?} less {suffix:?} settled to {settled:?}"
        );
        // Another text of the same byte length: one letter swapped for
        // another as long.
        let other = [
          ("a", "b"),
          ("b", "a"),
          ("日", "本"),
          ("本", "日"),
          ("語", "日"),
        ]
        .into_iter()
        .find(|(from, _)| text.contains(from))
        .map(|(from, to)| text.replacen(from, to, 1))
        .expect("the text holds a letter");
        assert_eq!(other.len(), text.len());
        let end = FieldEnd::cap(&text).with_cut(&suffix);
        assert_eq!(
          once_more(&other, end),
          (other.trim().into(), crate::DescriptionEnd::Unknown),
          "{other:?} under the account of {text:?} less {suffix:?}"
        );
      }
    }

    /// LAW: **`Task::parse_ended` settles the description by its
    /// `"description"` account exactly as `parse_with_description_end`
    /// settles it** — the model's end, the grammar's close at the cap, a
    /// named cut, and an account taken under another cap, refused by name —
    /// so an engine generic over `Task` reaches the same settling through the
    /// trait's door.
    #[test]
    fn parse_ended_settles_the_description_as_parse_with_description_end() {
      for written in AT_THE_CAP {
        let task = capped_at(written);
        let answer = answer(written);
        for end in [
          FieldEnd::model(written),
          FieldEnd::cap(written),
          FieldEnd::cap(written).with_cut(last_char(written)),
        ] {
          let mut ends = FieldEnds::new();
          ends.insert("description", end.clone());
          let through_the_door = task
            .parse_ended(&answer, &ends)
            .unwrap_or_else(|e| panic!("{written:?} under {end:?} must parse: {e:?}"));
          let direct = task
            .parse_with_description_end(&answer, &end)
            .unwrap_or_else(|e| panic!("{written:?} under {end:?} must parse: {e:?}"));
          assert_eq!(through_the_door, direct, "{written:?} under {end:?}");
          assert_ne!(
            through_the_door.description_end(),
            crate::DescriptionEnd::Unknown,
            "{written:?}: the account is read"
          );
        }
      }
      let task = ImageAnalysisTask::new();
      let answer = answer("abc\u{FFFD}");
      let end = FieldEnd::cap("abc\u{FFFD}").with_cut("\u{FFFD}");
      let mut ends = FieldEnds::new();
      ends.insert("description", end.clone());
      for refused in [
        task.parse_ended(&answer, &ends),
        task.parse_with_description_end(&answer, &end),
      ] {
        assert!(
          matches!(
            refused,
            Err(JsonParseError::DescriptionCapMismatch { cap: 120, chars: 4 })
          ),
          "an account taken under another cap: {refused:?}"
        );
      }
    }

    /// LAW: **an account of any other field changes nothing.** The
    /// description is the one string the task caps. With no `"description"`
    /// entry — no accounts at all, or only accounts of other fields: one the
    /// task asks for (`tags`), one it does not (`scene`), the description's
    /// name in another case or spelled as a JSON pointer, the empty name —
    /// `parse_ended` is `parse`, the description `Unknown`; beside a
    /// description account they change nothing that account settles.
    #[test]
    fn an_account_of_another_field_changes_nothing() {
      let written = AT_THE_CAP[0];
      let task = capped_at(written);
      let answer = answer(written);
      let parse_ended = |ends: &FieldEnds| {
        task
          .parse_ended(&answer, ends)
          .unwrap_or_else(|e| panic!("{ends:?} must parse: {e:?}"))
      };
      let parsed = task.parse(&answer).expect("the answer parses");
      assert_eq!(parsed.description_end(), crate::DescriptionEnd::Unknown);
      let mut ends = FieldEnds::new();
      assert_eq!(parse_ended(&ends), parsed, "no accounts");
      for (field, end) in [
        ("tags", FieldEnd::cap("label")),
        ("scene", FieldEnd::model("kitchen")),
        ("Description", FieldEnd::cap(written)),
        ("/description", FieldEnd::model(written)),
        ("", FieldEnd::cap(written)),
      ] {
        ends.insert(field, end);
        assert_eq!(parse_ended(&ends), parsed, "an account of {field:?}");
      }
      let end = FieldEnd::cap(written);
      ends.insert("description", end.clone());
      let direct = task
        .parse_with_description_end(&answer, &end)
        .expect("the answer parses with the description's account");
      assert_eq!(direct.description_end(), crate::DescriptionEnd::Ragged);
      assert_eq!(
        parse_ended(&ends),
        direct,
        "beside the description's account"
      );
    }

    /// LAW: **the task declares the description's cap exactly as its schema
    /// states it, and no other field's**, whatever the cap and the roster.
    #[test]
    fn the_task_declares_the_description_cap_its_schema_states() {
      for cap in [1, 7, 120, 500] {
        for task in [
          ImageAnalysisTask::new().with_description_max_chars(nz(cap)),
          full_task().with_description_max_chars(nz(cap)),
        ] {
          let caps = task.field_caps();
          assert_eq!(caps.len(), 1, "{cap}");
          assert_eq!(caps.get("description"), Some(cap));
          assert_eq!(
            task.schema()["properties"]["description"]["maxLength"],
            cap,
            "the declared cap is the schema's"
          );
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

    // ===== the list extensions' caps (llmtask#18) =====

    /// The six list extensions, in [`Extension::ALL`] order.
    const LISTS: [Extension; 6] = [
      Extension::Subjects,
      Extension::Objects,
      Extension::Actions,
      Extension::Emotion,
      Extension::Lighting,
      Extension::Categories,
    ];

    /// How the parsed analysis says `extension`'s list ended.
    fn list_end(analysis: &ImageAnalysis, extension: Extension) -> ListEnd {
      match extension {
        Extension::Subjects => analysis.subjects_end(),
        Extension::Objects => analysis.objects_end(),
        Extension::Actions => analysis.actions_end(),
        Extension::Emotion => analysis.emotion_end(),
        Extension::Lighting => analysis.lighting_end(),
        Extension::Categories => analysis.categories_end(),
        Extension::Scene | Extension::ShotType => panic!("{extension:?} is not a list"),
      }
    }

    /// The answer to a task that asks for `extension` besides the default
    /// fields, whose list is `labels`.
    fn list_answer(extension: Extension, labels: &[String]) -> String {
      let list = serde_json::to_string(labels).expect("strings serialize");
      format!(r#"{{{DEFAULT_MEMBERS},"{}":{list}}}"#, extension.as_str())
    }

    /// LAW (the schema golden): **every list extension states its cap in
    /// the schema as `maxItems`**, at its default unless a builder sets it:
    /// `subjects`, `objects` and `actions` at `tags`' 8, `emotion`,
    /// `lighting` and `categories` at 3. The full roster's schema, key for
    /// key, and each list extension's entry alone.
    #[test]
    fn every_list_extension_states_its_cap_in_the_schema() {
      let array =
        |cap: usize| json!({ "type": "array", "items": { "type": "string" }, "maxItems": cap });
      assert_eq!(
        *full_task().schema(),
        json!({
          "type": "object",
          "properties": {
            "scene": { "type": "string" },
            "description": { "type": "string", "maxLength": 120 },
            "subjects": array(8),
            "objects": array(8),
            "actions": array(8),
            "emotion": array(3),
            "shot_type": { "type": "string" },
            "lighting": array(3),
            "tags": array(8),
            "categories": array(3)
          },
          "required": [
            "scene", "description", "subjects", "objects", "actions",
            "emotion", "shot_type", "lighting", "tags", "categories"
          ],
          "additionalProperties": false
        })
      );
      for extension in LISTS {
        let cap = ImageAnalysisTask::default_max_items(extension).expect("a list is capped");
        let task = ImageAnalysisTask::new().with_extensions([extension]);
        assert_eq!(task.max_items(extension), Some(cap), "{extension:?}");
        assert_eq!(
          task.schema()["properties"][extension.as_str()],
          array(cap.get()),
          "{extension:?} alone"
        );
      }
    }

    /// LAW: **the schema uses no keyword beyond those an engine serving this
    /// task must honour** — `type`, `properties`, `required`,
    /// `additionalProperties`, `items`, `maxLength` and `maxItems` — on any
    /// roster and at any cap. An engine refuses a schema that uses a keyword
    /// it does not implement (llguidance implements no `uniqueItems`), so
    /// one more would fail every request instead of shaping the answer.
    #[test]
    fn the_schema_uses_only_the_keywords_an_engine_must_honour() {
      fn keywords<'s>(schema: &'s Value, found: &mut Vec<&'s str>) {
        let Some(members) = schema.as_object() else {
          return;
        };
        for (keyword, value) in members {
          found.push(keyword);
          match keyword.as_str() {
            "properties" => {
              for property in value.as_object().into_iter().flat_map(Map::values) {
                keywords(property, found);
              }
            }
            "items" => keywords(value, found),
            _ => {}
          }
        }
      }
      let honoured = [
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "maxLength",
        "maxItems",
      ];
      let tasks = [
        ImageAnalysisTask::new(),
        full_task(),
        full_task()
          .with_max_items(Extension::Categories, nz(50))
          .with_tags_max_items(nz(1)),
      ]
      .into_iter()
      .chain(LISTS.map(|extension| ImageAnalysisTask::new().with_extensions([extension])));
      for task in tasks {
        let mut found = Vec::new();
        keywords(task.schema(), &mut found);
        for keyword in found {
          assert!(
            honoured.contains(&keyword),
            "{keyword} is not a keyword the engine must honour"
          );
        }
      }
    }

    /// LAW: **a list's cap is its own.** `max_items` reads each list
    /// extension's default until `with_max_items` or `set_max_items` sets
    /// it; setting one changes that list's cap alone, in the schema too,
    /// and the cap stays while the extensions are switched; `scene` and
    /// `shot_type` have none, and setting one for them changes nothing.
    #[test]
    fn a_list_cap_is_set_per_extension() {
      let task = ImageAnalysisTask::new();
      for extension in Extension::ALL {
        assert_eq!(
          task.max_items(extension),
          ImageAnalysisTask::default_max_items(extension),
          "{extension:?}"
        );
      }
      for extension in LISTS {
        let raised = full_task().with_max_items(extension, nz(20));
        for other in Extension::ALL {
          let expected = if other == extension {
            Some(nz(20))
          } else {
            ImageAnalysisTask::default_max_items(other)
          };
          assert_eq!(
            raised.max_items(other),
            expected,
            "{extension:?} set, {other:?} read"
          );
        }
        assert_eq!(
          raised.schema()["properties"][extension.as_str()]["maxItems"],
          20
        );

        let mut task = ImageAnalysisTask::new().with_max_items(extension, nz(2));
        task.set_extensions([extension]);
        assert_eq!(
          task.schema()["properties"][extension.as_str()]["maxItems"],
          2
        );
        task.set_max_items(extension, nz(5));
        assert_eq!(task.max_items(extension), Some(nz(5)));
        assert_eq!(
          task.schema()["properties"][extension.as_str()]["maxItems"],
          5
        );
      }
      for string in [Extension::Scene, Extension::ShotType] {
        assert_eq!(ImageAnalysisTask::default_max_items(string), None);
        let task = full_task().with_max_items(string, nz(4));
        assert_eq!(task.max_items(string), None, "{string:?}");
        assert_eq!(
          task.schema(),
          full_task().schema(),
          "{string:?}: the schema is unchanged"
        );
      }
    }

    /// LAW (llmtask#18): **a list over its cap is read to its cap, `Capped`,
    /// never refused.** Fifty categories under a cap of 8 are the first 8 as
    /// written; every list extension, at its default cap and at others, keeps
    /// its first `cap` labels and drops the rest.
    #[test]
    fn a_list_over_its_cap_is_read_to_its_cap_and_capped() {
      let fifty: Vec<String> = (0..50).map(|n| format!("category {n}")).collect();
      let task = ImageAnalysisTask::new()
        .with_extensions([Extension::Categories])
        .with_max_items(Extension::Categories, nz(8));
      let analysis = task
        .parse(&list_answer(Extension::Categories, &fifty))
        .expect("a list over its cap is read to it");
      let first_eight: Vec<&str> = fifty[..8].iter().map(String::as_str).collect();
      assert_eq!(read(&analysis, Extension::Categories), first_eight);
      assert_eq!(analysis.categories_end(), ListEnd::Capped);

      for extension in LISTS {
        let default = ImageAnalysisTask::default_max_items(extension).expect("a list is capped");
        for cap in [default.get(), 1, 12] {
          let task = ImageAnalysisTask::new()
            .with_extensions([extension])
            .with_max_items(extension, nz(cap));
          let labels: Vec<String> = (0..cap + 7).map(|n| format!("label {n}")).collect();
          let analysis = task
            .parse(&list_answer(extension, &labels))
            .unwrap_or_else(|err| panic!("{extension:?} at {cap}: {err:?}"));
          let kept: Vec<&str> = labels[..cap].iter().map(String::as_str).collect();
          assert_eq!(read(&analysis, extension), kept, "{extension:?} at {cap}");
          assert_eq!(
            list_end(&analysis, extension),
            ListEnd::Capped,
            "{extension:?} at {cap}"
          );
        }
      }
    }

    /// LAW: **a list ends `Capped` once the answer lists its cap's count, and
    /// `Whole` short of it.** The cap counts the elements as the answer wrote
    /// them, as the schema's `maxItems` does, before labels are trimmed and
    /// deduplicated: a duplicate or a blank element takes its place under the
    /// cap.
    #[test]
    fn a_list_is_capped_at_its_cap_and_whole_short_of_it() {
      let cases: [(&[&str], &[&str], ListEnd); 9] = [
        (&[], &[], ListEnd::Whole),
        (&["a"], &["a"], ListEnd::Whole),
        (&["a", "b"], &["a", "b"], ListEnd::Whole),
        (&["a", "a"], &["a"], ListEnd::Whole),
        (&["a", "b", "c"], &["a", "b", "c"], ListEnd::Capped),
        (&["a", "b", "c", "d"], &["a", "b", "c"], ListEnd::Capped),
        (&["a", "a", "b"], &["a", "b"], ListEnd::Capped),
        (&["a", "a", "b", "c"], &["a", "b"], ListEnd::Capped),
        (&["", " ", "a", "b"], &["a"], ListEnd::Capped),
      ];
      for extension in LISTS {
        let task = ImageAnalysisTask::new()
          .with_extensions([extension])
          .with_max_items(extension, nz(3));
        for (written, kept, end) in cases {
          let written: Vec<String> = written.iter().map(|label| String::from(*label)).collect();
          let analysis = task
            .parse(&list_answer(extension, &written))
            .unwrap_or_else(|err| panic!("{extension:?} {written:?}: {err:?}"));
          assert_eq!(
            read(&analysis, extension),
            kept,
            "{extension:?} {written:?}"
          );
          assert_eq!(
            list_end(&analysis, extension),
            end,
            "{extension:?} {written:?}"
          );
        }
      }
    }

    /// LAW: **the bounded fields keep their rule.** `description` and `tags`
    /// state their caps as they did, an answer over either is refused by
    /// name — beside a list extension over its own cap, which is read to it —
    /// and an answer cut off inside or after either is not JSON.
    #[test]
    fn the_bounded_fields_keep_their_rule() {
      for task in [ImageAnalysisTask::new(), full_task()] {
        let task = task
          .with_description_max_chars(nz(10))
          .with_tags_max_items(nz(2));
        let properties = &task.schema()["properties"];
        assert_eq!(
          properties["description"],
          json!({ "type": "string", "maxLength": 10 })
        );
        assert_eq!(
          properties["tags"],
          json!({ "type": "array", "items": { "type": "string" }, "maxItems": 2 })
        );
      }

      let task = ImageAnalysisTask::new()
        .with_extensions([Extension::Categories])
        .with_tags_max_items(nz(2));
      let categories = r#""categories":["home","leisure","indoor","quiet"]"#;
      match task.parse(&format!(
        r#"{{"description":"A person reads.","tags":["reading","book","lamp"],{categories}}}"#
      )) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["tags"]),
        other => panic!("tags over its cap must be refused by name alone, got {other:?}"),
      }
      let analysis = task
        .parse(&format!(
          r#"{{"description":"A person reads.","tags":["reading","book"],{categories}}}"#
        ))
        .expect("tags at its cap and categories over theirs");
      assert_eq!(analysis.tags(), ["reading", "book"]);
      assert_eq!(analysis.categories(), ["home", "leisure", "indoor"]);
      assert_eq!(analysis.categories_end(), ListEnd::Capped);

      for cut in [
        r#"{"description":"A person re"#,
        r#"{"description":"A person reads."#,
        r#"{"description":"A person reads.","tags":["rea"#,
        r#"{"description":"A person reads.","tags":["reading","#,
        r#"{"description":"A person reads.","tags":["reading"]"#,
      ] {
        assert!(
          matches!(task.parse(cut), Err(JsonParseError::Json(_))),
          "{cut} is not JSON"
        );
      }
    }

    /// The answer `an_answer_cut_inside_a_list_keeps_the_items_written_whole`
    /// cuts: the full roster in field order — the order an engine writes the
    /// schema's members in — each list holding labels with escapes and a
    /// two-byte character.
    const WRITTEN: &str = r#"{"scene":"office","description":"People work at their desks.","subjects":["office worker","visitor \"guest\""],"objects":["desk","caf\u00e9 cup","lamp"],"actions":["typing"],"emotion":["calm","focused"],"shot_type":"wide","lighting":["daylight","é"],"tags":["office","work"],"categories":["work","busi\\ness"]}"#;

    /// One list extension's array in [`WRITTEN`]: the bytes of its `[` and
    /// its `]`, and each label it holds with the byte just past it.
    struct WrittenList {
      extension: Extension,
      open: usize,
      close: usize,
      labels: Vec<(String, usize)>,
    }

    /// Every list extension's array in [`WRITTEN`], read by serde_json.
    fn written_lists() -> Vec<WrittenList> {
      LISTS
        .into_iter()
        .map(|extension| {
          let open = WRITTEN
            .find(&format!(r#""{}":["#, extension.as_str()))
            .expect("the list is written")
            + extension.as_str().len()
            + 3;
          let mut stream =
            serde_json::Deserializer::from_str(&WRITTEN[open..]).into_iter::<Vec<&RawValue>>();
          let items = stream
            .next()
            .expect("an array")
            .expect("a JSON array of values");
          let close = open + stream.byte_offset() - 1;
          let labels = items
            .into_iter()
            .map(|item| {
              let start = item.get().as_ptr() as usize - WRITTEN.as_ptr() as usize;
              let label: String = serde_json::from_str(item.get()).expect("a string");
              (label, start + item.get().len())
            })
            .collect();
          WrittenList {
            extension,
            open,
            close,
            labels,
          }
        })
        .collect()
    }

    /// LAW (llmtask#18): **an answer cut off inside a list extension keeps
    /// the labels written whole; any other cut is refused.** Every prefix of
    /// a full-roster answer, cut at each character boundary:
    ///
    /// - inside `categories`' array, the answer's last field: it parses,
    ///   `categories` holding the labels written whole before the cut,
    ///   `Capped`, and every other field read as the whole answer reads it;
    /// - inside another list extension's array: the fields after it are
    ///   unwritten, `MissingFields` naming exactly those;
    /// - anywhere else — before the object opens, in a key or before its
    ///   value, inside or after a string field, inside or after `tags`, after
    ///   a list's own `]`: `Json`.
    #[test]
    fn an_answer_cut_inside_a_list_keeps_the_items_written_whole() {
      let task = full_task();
      let whole = task.parse(WRITTEN).expect("the whole answer parses");
      assert_eq!(whole.categories(), ["work", "busi\\ness"]);
      assert_eq!(whole.categories_end(), ListEnd::Whole);
      let lists = written_lists();
      // How many cuts each rule took: read, refused by name, not JSON.
      let mut taken = [0_usize; 3];
      let cuts = (0..WRITTEN.len()).filter(|&cut| WRITTEN.is_char_boundary(cut));
      for cut in cuts {
        let answer = &WRITTEN[..cut];
        let parsed = task.parse(answer);
        match lists
          .iter()
          .find(|list| list.open < cut && cut <= list.close)
        {
          Some(list) if list.extension == Extension::Categories => {
            taken[0] += 1;
            let analysis = parsed.unwrap_or_else(|err| panic!("{answer:?} must parse: {err:?}"));
            let kept: Vec<&str> = list
              .labels
              .iter()
              .filter(|(_, end)| *end <= cut)
              .map(|(label, _)| label.as_str())
              .collect();
            assert_eq!(read(&analysis, Extension::Categories), kept, "{answer:?}");
            assert_eq!(analysis.categories_end(), ListEnd::Capped, "{answer:?}");
            let restored = analysis
              .with_categories(whole.categories().to_vec())
              .with_categories_end(whole.categories_end());
            assert_eq!(restored, whole, "{answer:?}: every other field as written");
          }
          Some(list) => {
            taken[1] += 1;
            let unwritten: Vec<&str> = Field::ALL
              .into_iter()
              .skip_while(|field| field.key() != list.extension.as_str())
              .skip(1)
              .map(Field::key)
              .collect();
            match parsed {
              Err(JsonParseError::MissingFields(fields)) => {
                assert_eq!(fields, unwritten, "{answer:?}")
              }
              other => panic!("{answer:?} must name the unwritten fields, got {other:?}"),
            }
          }
          None => {
            taken[2] += 1;
            assert!(
              matches!(parsed, Err(JsonParseError::Json(_))),
              "{answer:?} is not JSON, got {parsed:?}"
            );
          }
        }
      }
      assert!(
        taken.iter().all(|&cuts| cuts > 0),
        "every rule takes a cut: {taken:?}"
      );
    }

    /// LAW (llmtask#18): **a list cut mid-item keeps the items before it.**
    /// `categories` cut inside its third label keeps the two before it,
    /// `Capped`, and the answer's other fields read as written.
    #[test]
    fn a_list_cut_mid_item_keeps_the_items_before_it() {
      let task = ImageAnalysisTask::new().with_extensions([Extension::Categories]);
      let analysis = task
        .parse(r#"{"description":"A person reads by a window.","tags":["reading"],"categories":["home","leisure","indo"#)
        .expect("a cut inside the last list is read to its whole items");
      assert_eq!(analysis.categories(), ["home", "leisure"]);
      assert_eq!(analysis.categories_end(), ListEnd::Capped);
      assert_eq!(analysis.description(), "A person reads by a window.");
      assert_eq!(analysis.tags(), ["reading"]);
    }

    /// LAW: **a cut answer is closed, then held to every check; a cut it
    /// cannot be closed at is not JSON.** Closed after the cut list's last
    /// whole item, an answer that leaves a field it asks for unwritten is
    /// `MissingFields` naming it; an undeclared key, a duplicated key or a
    /// label of another type before the cut is refused as in any answer. A
    /// cut inside an item that is not a string, after a list's own `]`, in a
    /// key, inside `tags`, inside the description, or inside a list the task
    /// does not ask for, is not JSON; and so is a cut answer that is not
    /// JSON before the cut.
    #[test]
    fn a_cut_answer_is_refused_by_name_or_is_not_json() {
      let task =
        ImageAnalysisTask::new().with_extensions([Extension::Subjects, Extension::Categories]);
      let head = r#"{"description":"A person reads.","tags":["reading"],"subjects":[],"#;
      match task.parse(r#"{"description":"A person reads.","subjects":["reader","ca"#) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["tags", "categories"]),
        other => panic!("expected the unwritten fields named, got {other:?}"),
      }
      match task.parse(&format!(r#"{head}"extra":1,"categories":["ho"#)) {
        Err(JsonParseError::UnknownFields(fields)) => assert_eq!(fields, ["extra"]),
        other => panic!("expected UnknownFields naming extra, got {other:?}"),
      }
      match task.parse(&format!(r#"{head}"categories":["home"],"categories":["ho"#)) {
        Err(JsonParseError::DuplicateField(key)) => assert_eq!(key, "categories"),
        other => panic!("expected DuplicateField naming categories, got {other:?}"),
      }
      match task.parse(&format!(r#"{head}"categories":["home",7,"ind"#)) {
        Err(JsonParseError::MissingFields(fields)) => assert_eq!(fields, ["categories"]),
        other => panic!("expected MissingFields naming categories, got {other:?}"),
      }
      for cut in [
        format!(r#"{head}"categories":["home",7"#),
        format!(r#"{head}"categories":["home",tr"#),
        format!(r#"{head}"categories":["home",["in"#),
        format!(r#"{head}"categories":["home"]"#),
        format!(r#"{head}"categ"#),
        format!(r#"{head}"categories""#),
        format!(r#"{head}"categories":"#),
        String::from(r#"{"description":"A person reads.","tags":["rea"#),
        String::from(r#"{"description":"A person re"#),
        format!(r#"{head}"categories":["\uD800","ho"#),
        String::from(r#"["home","#),
      ] {
        assert!(
          matches!(task.parse(&cut), Err(JsonParseError::Json(_))),
          "{cut} is not JSON"
        );
      }
      assert!(
        matches!(
          ImageAnalysisTask::new()
            .parse(r#"{"description":"A person reads.","tags":["reading"],"categories":["ho"#),
          Err(JsonParseError::Json(_))
        ),
        "a list the task does not ask for is not read"
      );
    }

    /// LAW: **a list's end needs no account.** `parse_ended` and
    /// `parse_with_description_end` read every list as `parse` does — under,
    /// at and over its cap, and cut off inside it — whatever accounts they
    /// are handed.
    #[test]
    fn the_decoders_account_changes_no_list() {
      let task = full_task().with_max_items(Extension::Objects, nz(2));
      let mut ends = FieldEnds::new();
      ends.insert(
        "description",
        FieldEnd::model("People work at their desks."),
      );
      ends.insert("categories", FieldEnd::cap("work"));
      let description = FieldEnd::model("People work at their desks.");
      for answer in [WRITTEN, &WRITTEN[..WRITTEN.len() - 10]] {
        let parsed = task.parse(answer).expect("the answer parses");
        let ended = task.parse_ended(answer, &ends).expect("the answer parses");
        let settled = task
          .parse_with_description_end(answer, &description)
          .expect("the answer parses");
        for extension in LISTS {
          for other in [&ended, &settled] {
            assert_eq!(
              read(other, extension),
              read(&parsed, extension),
              "{extension:?}"
            );
            assert_eq!(
              list_end(other, extension),
              list_end(&parsed, extension),
              "{extension:?}"
            );
          }
        }
        assert_eq!(
          parsed.objects_end(),
          ListEnd::Capped,
          "three objects under a cap of 2"
        );
      }
    }
  }
}
