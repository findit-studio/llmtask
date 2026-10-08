//! `Task` trait and parse-error types — the cross-engine
//! abstraction — and a constrained decoder's account of how it ended
//! an answer's string fields ([`FieldEnd`], [`FieldEnds`]), which an
//! engine hands a task through [`Task::parse_ended`], read against the
//! caps the task declares ([`FieldCaps`], [`Task::field_caps`]).
//!
//! `Task::ParseError` is an associated type so the trait does not
//! bake in any concrete error representation. Tasks that parse
//! JSON output use [`crate::JsonParseError`] (gated on the `json`
//! feature); Tasks that parse free-form text or custom formats
//! choose their own error type.

use smol_str::SmolStr;
// `BTreeMap` under both std (resolves via the `extern crate std`) and
// alloc-only (resolves via the `extern crate alloc as std` alias in
// lib.rs).
use std::collections::BTreeMap;

use crate::grammar::Grammar;

/// A structured-output task description.
///
/// Implementations supply the prompt, the constrained-decoding
/// [`Grammar`], and a parser that turns the model's raw text into
/// a typed `Output`.
///
/// **No thread-safety bounds at the trait level.** `Task` itself
/// is unbounded so non-`Send` implementors (e.g., a Task carrying
/// an `Rc` for setup-time state) compile without ceremony.
/// Engines that spawn parse work across threads (e.g., a tokio-
/// driven inference server) add `Send + Sync + 'static` bounds at
/// their generic call sites, on `T`, `T::Output`, and
/// `T::ParseError`. The Rust pattern: bound where you need, not
/// where you might.
///
/// Implementations should cache their grammar (build it once in
/// `new`) rather than rebuilding it per call.
///
/// # Required methods
///
/// Implementors provide these four methods explicitly. The trait
/// used to expose only `schema(&self) -> &serde_json::Value` and a
/// fixed `ParseError` enum, but both have been generalized:
///
/// - `schema(&self) -> &Self::Value` — borrow the typed schema
///   (engines that bind `Value = serde_json::Value` get typed
///   access without going through the [`Grammar`] enum).
/// - `grammar(&self) -> Grammar` — wrap the schema in the
///   engine-agnostic enum (engines that handle multiple variants
///   pattern-match on this).
/// - `parse(&self, raw: &str) -> Result<Self::Output, Self::ParseError>` —
///   typed parse step.
/// - `prompt(&self) -> &str` — the user-message prompt.
///
/// # Provided methods
///
/// - `parse_ended(&self, raw: &str, ends: &FieldEnds) -> Result<Self::Output, Self::ParseError>` —
///   the parse step with a constrained decoder's account of how it
///   ended each string field. The default ignores the accounts and
///   returns [`Task::parse`]'s result; a task that settles a capped
///   field by its account overrides it.
/// - `field_caps(&self) -> FieldCaps` — the `maxLength` the task's
///   grammar puts on each top-level string field, which an engine reads
///   those accounts against. The default declares none; a task that
///   settles a capped field by its account declares that field's cap.
///
/// Tasks that parse JSON output typically set
/// `type ParseError = llmtask::JsonParseError;` (the convenience
/// type behind the `json` feature). Custom Tasks pick any error
/// type that's `core::error::Error`.
pub trait Task {
  /// The typed result of a successful run.
  type Output;

  /// The schema/grammar value the Task carries. Typically:
  ///
  /// - `serde_json::Value` for JSON Schema tasks
  /// - `smol_str::SmolStr` for Lark / Regex string-grammar tasks
  /// - any other type the Task wants to expose as its schema
  ///   representation
  ///
  /// Engines that handle ONE specific schema type can bind it
  /// directly: `fn run<T: Task<Value = serde_json::Value>>(...)`
  /// — `task.schema()` then returns the typed value without an
  /// enum match. Engines that handle multiple schema types use
  /// `task.grammar()` and pattern-match on the [`Grammar`] enum.
  type Value;

  /// The error type returned by [`Task::parse`]. JSON-parsing
  /// Tasks typically use [`crate::JsonParseError`] (behind the
  /// `json` feature).
  type ParseError: core::error::Error;

  /// The user-message prompt sent alongside the images.
  fn prompt(&self) -> &str;

  /// Borrow the schema/grammar value. Zero-cost typed access for
  /// engines that bind [`Task::Value`] to a concrete type. Engines
  /// that need a unified [`Grammar`] enum should call
  /// [`Task::grammar`] instead.
  ///
  /// Pair `schema()` with `grammar()` when implementing: cache the
  /// schema once on the Task struct, return a borrow from
  /// `schema()`, build the [`Grammar`] wrapper in `grammar()`.
  fn schema(&self) -> &Self::Value;

  /// Constrained-decoding grammar for this task, wrapped in the
  /// engine-agnostic [`Grammar`] enum.
  ///
  /// Always required (no default impl). Implementations are
  /// typically a one-liner that wraps `self.schema()` in the
  /// appropriate variant — for example:
  ///
  /// ```ignore
  /// // JSON task:
  /// fn grammar(&self) -> Grammar {
  ///     Grammar::JsonSchema(self.schema().clone())
  /// }
  ///
  /// // Lark task:
  /// fn grammar(&self) -> Grammar {
  ///     Grammar::Lark(self.schema().clone())
  /// }
  /// ```
  ///
  /// A default impl was considered but rejected: the bound it
  /// would have required (`Self::Value: Clone + Into<Grammar>`)
  /// also gets checked at every call site, so Tasks whose
  /// `Value` doesn't satisfy the bound (e.g., `SmolStr`, which
  /// is ambiguous between Lark and Regex) couldn't have their
  /// `grammar()` called even when overridden. Two-line override
  /// per Task is the simpler shape.
  fn grammar(&self) -> Grammar;

  /// Parse the model's raw text output into a typed `Output`.
  fn parse(&self, raw: &str) -> Result<Self::Output, Self::ParseError>;

  /// [`Task::parse`], with a constrained decoder's account of how it
  /// ended the answer's string fields.
  ///
  /// An engine that decodes under the task's grammar knows, at the step
  /// it closes a string field, whether the model closed it or the grammar
  /// closed it at the field's `maxLength`; the answer's text does not
  /// carry that. Such an engine calls this in place of [`Task::parse`],
  /// handing over in `ends` an account of every string field it took one
  /// for. A task reads the accounts of the fields it caps and ignores the
  /// rest; a field with no account is parsed as [`Task::parse`] parses it.
  ///
  /// The default ignores `ends` and returns [`Task::parse`]'s result, so
  /// an engine can call this on every task, and a task that reads no
  /// account needs no override.
  ///
  /// # Errors
  ///
  /// As [`Task::parse`]; an override may also refuse an account that
  /// contradicts the field it describes.
  fn parse_ended(&self, raw: &str, ends: &FieldEnds) -> Result<Self::Output, Self::ParseError> {
    let _ = ends;
    self.parse(raw)
  }

  /// The `maxLength` this task's grammar puts on each of the answer's
  /// top-level string fields, in Unicode scalar values, by field name.
  ///
  /// An engine reads the account of how it ended a field against the cap
  /// declared here: a field that closed holding exactly its declared cap was
  /// bound by it, whichever token carried the closing quote, and its account
  /// is [`FieldEnd::cap`]. The engine reads no cap out of the grammar: the
  /// task that wrote the grammar is the one authority on its caps.
  ///
  /// **A task that caps a string field in its grammar but does not declare
  /// the cap here gets no `cap` account for that field.** A string the model
  /// closed is still [`FieldEnd::model`], and one the grammar closed at the
  /// cap has no account at all. A task that settles a capped field by its
  /// account declares that field's cap, exactly as its grammar states it.
  ///
  /// The default declares none.
  fn field_caps(&self) -> FieldCaps {
    FieldCaps::new()
  }
}

// ===== the decoder's account of how it ended a string field =====

/// How a constrained decoder ended one string field of an answer. An
/// engine that decodes under the task's grammar knows it at the step it
/// closes the string; the answer's text does not carry it. The engine
/// hands a task its accounts, keyed by field ([`FieldEnds`]), through
/// [`Task::parse_ended`]; `image_analysis::ImageAnalysisTask` settles its
/// description by one, which `parse_with_description_end` also takes
/// directly.
///
/// # One string
///
/// Every account is about ONE string: the field exactly as the answer
/// carries it — JSON-decoded, untrimmed — which it holds whole
/// ([`Self::source`]) and is bound to byte for byte. A field that is not
/// exactly that string is not one the account describes, and nothing is
/// known about how it ends: a settled field, shorter by a suffix or by its
/// trimming; another string of the same length; an account left from a
/// retry, or taken for another answer in a batch. `ImageAnalysisTask`
/// removes nothing from such a description and marks it
/// [`DescriptionEnd::Unknown`](crate::DescriptionEnd::Unknown).
///
/// # What an engine reports
///
/// Whether the model closed the string ([`Self::model`]) or the grammar
/// closed it at its `maxLength` ([`Self::cap`]); and, when the string's
/// last token was cut — its bytes an incomplete sequence the detokenizer
/// decoded as U+FFFD — that token's text ([`Self::with_cut`]). The account
/// names the suffix by its BYTES, never by a position: a position moves
/// when the text is trimmed. A flag saying only *that* a token was cut is
/// not an account of which bytes: with none, nothing is removed.
///
/// # A cap's invariant
///
/// The grammar closes a string exactly at its cap, so the string a cap's
/// account describes holds exactly as many characters (Unicode scalar
/// values, as JSON Schema's `maxLength` counts them) as the field's cap.
/// One that does not was taken under another cap: the account and the task
/// disagree, a configuration skew a task refuses by name rather than
/// settles — `ImageAnalysisTask` as
/// `JsonParseError::DescriptionCapMismatch`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldEnd {
  closed_at_cap: bool,
  source: SmolStr,
  cut: Option<SmolStr>,
}

impl FieldEnd {
  /// The model closed `field` itself: the grammar would have let it go
  /// on. `field` is the string exactly as the answer carries it —
  /// JSON-decoded, untrimmed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn model(field: &str) -> Self {
    Self {
      closed_at_cap: false,
      source: SmolStr::new(field),
      cut: None,
    }
  }

  /// The grammar closed `field` at its `maxLength`: the model was left no
  /// choice but to end it there. `field` is the string exactly as the
  /// answer carries it — JSON-decoded, untrimmed — and holds exactly the
  /// cap's characters; only a capped string carries this account.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn cap(field: &str) -> Self {
    Self {
      closed_at_cap: true,
      source: SmolStr::new(field),
      cut: None,
    }
  }

  /// Whether the grammar closed the string at the field's `maxLength`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn closed_at_cap(&self) -> bool {
    self.closed_at_cap
  }

  /// The one string this account describes, exactly as the answer
  /// carries it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn source(&self) -> &str {
    &self.source
  }

  /// The cut last token's text, when the decoder named one — see
  /// [`Self::with_cut`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn cut(&self) -> Option<&str> {
    self.cut.as_deref()
  }

  /// Builder-style setter for [`Self::cut`]: the last token of the
  /// account's string was cut, and `suffix` is its text, exactly as the
  /// string ends with it — the detokenizer's U+FFFD included.
  ///
  /// A field that is the account's string, closed at the cap, loses
  /// exactly those bytes, and only when it ends with them and keeps text
  /// before them; anything else (a suffix the text does not end with, an
  /// empty one, one that is the whole text) removes nothing and keeps the
  /// trimmed text. Only the grammar's close cuts a token, so a
  /// [`Self::model`] account ignores it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[must_use]
  pub fn with_cut(mut self, suffix: &str) -> Self {
    self.cut = Some(SmolStr::new(suffix));
    self
  }
}

/// A constrained decoder's accounts of how it ended an answer's string
/// fields, one [`FieldEnd`] per field: what an engine hands
/// [`Task::parse_ended`].
///
/// A field is named by its key in the answer's top-level JSON object,
/// JSON-decoded (`"description"`). A field with no entry has no account:
/// how it ended is unknown.
///
/// ```
/// use llmtask::{FieldEnd, FieldEnds};
///
/// let mut ends = FieldEnds::new();
/// assert!(ends.is_empty());
/// ends.insert("description", FieldEnd::cap("A cat sleeps on a"));
/// ends.insert("scene", FieldEnd::model("kitchen"));
/// assert_eq!(ends.len(), 2);
/// assert!(ends.get("description").is_some_and(FieldEnd::closed_at_cap));
/// assert!(ends.get("tags").is_none());
/// // In field-name order.
/// let fields: Vec<&str> = ends.iter().map(|(field, _)| field).collect();
/// assert_eq!(fields, ["description", "scene"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldEnds {
  ends: BTreeMap<SmolStr, FieldEnd>,
}

impl FieldEnds {
  /// No accounts.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      ends: BTreeMap::new(),
    }
  }

  /// Records `end` as the account of `field`, returning the account it
  /// replaces, if `field` had one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn insert(&mut self, field: impl Into<SmolStr>, end: FieldEnd) -> Option<FieldEnd> {
    self.ends.insert(field.into(), end)
  }

  /// The account of `field`, if it has one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn get(&self, field: &str) -> Option<&FieldEnd> {
    self.ends.get(field)
  }

  /// Every field's account, in field-name order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn iter(&self) -> impl Iterator<Item = (&str, &FieldEnd)> {
    self.ends.iter().map(|(field, end)| (field.as_str(), end))
  }

  /// How many fields have an account.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn len(&self) -> usize {
    self.ends.len()
  }

  /// Whether no field has an account.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn is_empty(&self) -> bool {
    self.ends.is_empty()
  }
}

impl Default for FieldEnds {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

/// The `maxLength` a task's grammar puts on each of the answer's top-level
/// string fields, in Unicode scalar values (what JSON Schema's `maxLength`
/// counts): what [`Task::field_caps`] declares.
///
/// A field is named as [`FieldEnds`] names it: by its key in the answer's
/// top-level JSON object, JSON-decoded (`"description"`). A field with no
/// entry declares no cap.
///
/// ```
/// use llmtask::FieldCaps;
///
/// let mut caps = FieldCaps::new();
/// assert!(caps.is_empty());
/// caps.insert("description", 120);
/// assert_eq!(caps.get("description"), Some(120));
/// assert_eq!(caps.get("scene"), None);
/// assert_eq!(caps.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldCaps {
  caps: BTreeMap<SmolStr, usize>,
}

impl FieldCaps {
  /// No caps.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      caps: BTreeMap::new(),
    }
  }

  /// Declares `cap` as the `maxLength` of `field`, returning the cap it
  /// replaces, if `field` had one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn insert(&mut self, field: impl Into<SmolStr>, cap: usize) -> Option<usize> {
    self.caps.insert(field.into(), cap)
  }

  /// The cap declared for `field`, if any.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn get(&self, field: &str) -> Option<usize> {
    self.caps.get(field).copied()
  }

  /// Every declared cap, in field-name order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn iter(&self) -> impl Iterator<Item = (&str, usize)> {
    self.caps.iter().map(|(field, cap)| (field.as_str(), *cap))
  }

  /// How many fields declare a cap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn len(&self) -> usize {
    self.caps.len()
  }

  /// Whether no field declares a cap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn is_empty(&self) -> bool {
    self.caps.is_empty()
  }
}

impl Default for FieldCaps {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

// ===== JSON parse error (json feature only) =====

#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
pub use json::JsonParseError;

#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
mod json {
  // Bring `Vec` into scope under both std (resolves via the
  // `extern crate std`) and alloc-only (resolves via the
  // `extern crate alloc as std` alias in lib.rs).
  use std::vec::Vec;

  use smol_str::SmolStr;

  /// Convenience parse-error type for [`crate::Task`]
  /// implementations whose model output is JSON. Available behind
  /// the `json` feature.
  ///
  /// Tasks set `type ParseError = llmtask::JsonParseError;`
  /// in their `impl Task` block; engines surface this via their
  /// own crate-level `Error::Parse(#[from] JsonParseError)`
  /// variant.
  #[derive(thiserror::Error, Debug)]
  pub enum JsonParseError {
    /// The response is not a JSON text: `serde_json` failed to parse it.
    /// `image_analysis::ImageAnalysisTask` also refuses here a string that
    /// escapes one half of a UTF-16 surrogate pair, which serde_json's
    /// grammar admits but no Unicode text can hold, wherever it sits.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The top-level JSON object declared the same member name more than
    /// once (e.g. `{"categories": null, "categories": []}`). A stock
    /// `serde_json::Value` decode collapses a duplicate object member
    /// silently — later overwrites earlier via `Map::insert` — before any
    /// of this crate's validation runs, so a well-typed copy of a field
    /// can mask an earlier null/wrong-typed copy (or the reverse,
    /// depending only on which copy comes last in the text), and a
    /// duplicate of a name [`Self::UnknownFields`] would otherwise catch
    /// can no longer be seen once collapsed to one entry. Tasks that
    /// read their top-level object through a duplicate-checking parse
    /// (e.g. `image_analysis::ImageAnalysisTask::parse`) return this
    /// instead, naming the repeated key, before any member's value is
    /// checked.
    #[error("schema violation: top-level key appears more than once: {0:?}")]
    DuplicateField(SmolStr),
    /// JSON parsed but one or more schema fields are unusable: a required
    /// field absent or present as JSON `null`, any listed field (required
    /// or optional) present with any JSON type other than the one its
    /// schema entry declares, or a value over a cap its schema entry
    /// declares (`maxLength`, `maxItems`). Another type means any of them:
    /// a number, a boolean or an object; a string where an array of strings
    /// is declared (a comma-separated list included); an array where a
    /// string is declared (a one-element array included); or an array
    /// holding anything but strings, whatever a number's magnitude or a
    /// nested value's depth. All four cases are folded into one
    /// variant because the schema requires every listed field to carry a
    /// value of its declared type within its caps — never null, never
    /// another type, never longer than declared — so a decoder that
    /// violates any of them has drifted the same way from the caller's
    /// perspective: the field's value can't be used. A task names such a
    /// field here; it never coerces the value into the declared shape and
    /// never truncates it to fit.
    #[error("schema violation: required fields missing, null, or invalid: {0:?}")]
    MissingFields(Vec<&'static str>),
    /// JSON parsed as an object, but it carries one or more keys outside
    /// the Task's declared `properties` — the runtime counterpart of the
    /// schema's `additionalProperties: false`. Distinct from
    /// [`Self::MissingFields`]: that variant names DECLARED fields whose
    /// *value* is absent, null, or wrong-typed, using `&'static str`
    /// because the declared field names are known at compile time. An
    /// unknown key is, by definition, not one of those names — it's
    /// arbitrary decoder output — so it can't borrow a `'static` name and
    /// is carried as an owned [`SmolStr`] instead.
    #[error("schema violation: object has fields outside the declared schema: {0:?}")]
    UnknownFields(Vec<SmolStr>),
    /// JSON parsed and had no missing fields, but every value was empty.
    #[error("structured response had no usable fields")]
    NoUsableFields,
    /// A constrained decoder's account says the grammar closed the
    /// description at its `maxLength`, but the description it describes
    /// does not hold exactly as many characters as this task's cap: the
    /// account was taken under another cap. A configuration skew between
    /// the engine and the task, refused rather than settled — see
    /// `image_analysis::FieldEnd`.
    #[error(
      "the decoder's account says the grammar closed the description at {chars} characters, \
       but this task's cap is {cap}"
    )]
    DescriptionCapMismatch {
      /// This task's `description_max_chars`.
      cap: usize,
      /// The characters (Unicode scalar values) of the description the
      /// account describes.
      chars: usize,
    },
  }
}

// Tests use `std::sync::OnceLock` (for static schema caching) and
// `format!` / `String` from the `std` prelude — gate them behind
// the `std` feature so the no_std + alloc build still typechecks
// against the lib code without dragging std into test compilation.
// The further `any(json, regex)` gate avoids `unused_imports`
// warnings (treated as errors in CI) for builds that turn `std`
// on without either of the test-bearing features.
#[cfg(all(test, feature = "std", any(feature = "json", feature = "regex")))]
mod tests {
  use super::*;
  // Only the json tests use OnceLock; gating the import keeps
  // `--features std,regex` (no json) free of unused-import warnings.
  #[cfg(feature = "json")]
  use std::sync::OnceLock;

  /// `Task` is dyn-compatible with `Output` and `ParseError`
  /// carrying through — note that `Value` cannot appear in the
  /// trait-object type list (not used in object-safe methods).
  #[cfg(feature = "json")]
  #[test]
  fn task_is_dyn_compatible() {
    struct Dummy;
    impl Task for Dummy {
      type Output = ();
      type Value = serde_json::Value;
      type ParseError = JsonParseError;
      fn prompt(&self) -> &str {
        ""
      }
      fn schema(&self) -> &serde_json::Value {
        static V: OnceLock<serde_json::Value> = OnceLock::new();
        V.get_or_init(|| serde_json::Value::Null)
      }
      fn grammar(&self) -> Grammar {
        Grammar::JsonSchema(self.schema().clone())
      }
      fn parse(&self, _raw: &str) -> Result<(), JsonParseError> {
        Ok(())
      }
    }
    let _: Box<dyn Task<Output = (), Value = serde_json::Value, ParseError = JsonParseError>> =
      Box::new(Dummy);
    fn _assert_send_sync(_: &impl ?Sized) {}
    _assert_send_sync(&*Box::new(Dummy)
      as &dyn Task<Output = (), Value = serde_json::Value, ParseError = JsonParseError>);
  }

  /// JSON Tasks get `grammar()` for free via the default impl.
  /// `Self::Value = serde_json::Value` satisfies
  /// `Clone + Into<Grammar>` (the From impl in this crate).
  #[cfg(feature = "json")]
  #[test]
  fn json_task_default_grammar_wraps_schema() {
    struct JsonTask;
    impl Task for JsonTask {
      type Output = ();
      type Value = serde_json::Value;
      type ParseError = JsonParseError;
      fn prompt(&self) -> &str {
        ""
      }
      fn schema(&self) -> &serde_json::Value {
        static V: OnceLock<serde_json::Value> = OnceLock::new();
        V.get_or_init(|| serde_json::json!({"type": "string"}))
      }
      fn grammar(&self) -> Grammar {
        Grammar::JsonSchema(self.schema().clone())
      }
      fn parse(&self, _raw: &str) -> Result<(), JsonParseError> {
        Ok(())
      }
    }
    let g = JsonTask.grammar();
    assert!(g.is_json_schema());
    assert_eq!(
      g.as_json_schema().unwrap(),
      &serde_json::json!({"type": "string"})
    );
  }

  /// Regex-only Task (no JSON dep). Demonstrates that the trait
  /// works without any JSON involvement when the consumer declares
  /// its own `Value` (the compiled `regex::Regex`), `ParseError`,
  /// and `grammar()`. Available behind the `regex` feature.
  #[cfg(feature = "regex")]
  #[test]
  fn regex_only_task_compiles_without_json_paths() {
    #[derive(Debug)]
    struct StringErr(String);
    impl std::fmt::Display for StringErr {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
      }
    }
    impl std::error::Error for StringErr {}

    struct TimestampTask {
      // Source pattern as a string — the canonical input we want
      // engines to receive (anchor-implicit / full-match semantics).
      pattern: smol_str::SmolStr,
      // Cached `Grammar` so `parse` can call `is_regex_full_match`
      // without rebuilding the grammar (and recompiling the regex)
      // on every call. `Grammar::is_regex_full_match` is the
      // engine-parity validator: full-match like the engine,
      // syntax-preserving for arbitrary regex (verbose mode,
      // alternation, etc.).
      grammar: Grammar,
    }
    impl Task for TimestampTask {
      type Output = String;
      // `Value = SmolStr` keeps the source pattern as the canonical
      // schema — what engines like llguidance consume — rather than
      // the compiled regex.
      type Value = smol_str::SmolStr;
      type ParseError = StringErr;
      fn prompt(&self) -> &str {
        "Output a date in YYYY-MM-DD format."
      }
      fn schema(&self) -> &smol_str::SmolStr {
        &self.pattern
      }
      fn grammar(&self) -> Grammar {
        self.grammar.clone()
      }
      fn parse(&self, raw: &str) -> Result<String, StringErr> {
        let trimmed = raw.trim();
        // `Some(true)` → full match. `Some(false)` → partial /
        // no match. `None` would mean grammar isn't Regex, which
        // can't happen here.
        if self.grammar.is_regex_full_match(trimmed) != Some(true) {
          return Err(StringErr(format!(
            "output {trimmed:?} does not match pattern {:?}",
            self.pattern.as_str()
          )));
        }
        Ok(trimmed.to_string())
      }
    }
    let pattern = smol_str::SmolStr::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}");
    let task = TimestampTask {
      grammar: Grammar::regex(&pattern).unwrap(),
      pattern,
    };
    assert_eq!(task.grammar().kind(), "regex");
    assert_eq!(task.schema().as_str(), r"[0-9]{4}-[0-9]{2}-[0-9]{2}");
    assert_eq!(task.parse("2026-05-09\n").unwrap(), "2026-05-09");
    assert!(task.parse("not a date").is_err());
    // Full-match validation rejects substrings the engine grammar
    // wouldn't accept.
    assert!(task.parse("abc2026-05-09xyz").is_err());
  }

  /// Engines that ONLY handle JSON Schema bind `Value =
  /// serde_json::Value` and skip the enum dispatch. This test
  /// shows that pattern works at compile time.
  #[cfg(feature = "json")]
  #[test]
  fn engine_can_bind_value_to_json_for_typed_access() {
    fn json_only_engine<T>(task: &T) -> &serde_json::Value
    where
      T: Task<Value = serde_json::Value>,
    {
      task.schema()
    }
    struct X;
    impl Task for X {
      type Output = ();
      type Value = serde_json::Value;
      type ParseError = JsonParseError;
      fn prompt(&self) -> &str {
        ""
      }
      fn schema(&self) -> &serde_json::Value {
        static V: OnceLock<serde_json::Value> = OnceLock::new();
        V.get_or_init(|| serde_json::json!({"type": "object"}))
      }
      fn grammar(&self) -> Grammar {
        Grammar::JsonSchema(self.schema().clone())
      }
      fn parse(&self, _raw: &str) -> Result<(), JsonParseError> {
        Ok(())
      }
    }
    let v = json_only_engine(&X);
    assert_eq!(v, &serde_json::json!({"type": "object"}));
  }
}

// Nothing here needs std or JSON, so these run under every feature set
// that compiles this module, `alloc` alone included.
#[cfg(test)]
mod field_ends_tests {
  use super::*;
  use std::vec::Vec;

  /// A Lark task that parses any non-empty answer to its length in bytes.
  struct Measure {
    grammar: SmolStr,
  }

  /// [`Measure`]'s refusal of an empty answer.
  #[derive(Debug, PartialEq, Eq)]
  struct EmptyAnswer;

  impl core::fmt::Display for EmptyAnswer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.write_str("the answer is empty")
    }
  }

  impl core::error::Error for EmptyAnswer {}

  impl Task for Measure {
    type Output = usize;
    type Value = SmolStr;
    type ParseError = EmptyAnswer;
    fn prompt(&self) -> &str {
      "Answer with any text."
    }
    fn schema(&self) -> &SmolStr {
      &self.grammar
    }
    fn grammar(&self) -> Grammar {
      Grammar::lark(self.grammar.clone())
    }
    fn parse(&self, raw: &str) -> Result<usize, EmptyAnswer> {
      if raw.is_empty() {
        Err(EmptyAnswer)
      } else {
        Ok(raw.len())
      }
    }
  }

  /// LAW: **the provided `parse_ended` is `parse`**, whatever accounts it
  /// is handed — none, or accounts of fields the answer does and does not
  /// carry — on an answer `parse` holds and on one it refuses.
  #[test]
  fn the_default_parse_ended_is_parse() {
    let task = Measure {
      grammar: SmolStr::new("start: /.*/"),
    };
    let mut ends = FieldEnds::new();
    let none = ends.clone();
    ends.insert("description", FieldEnd::cap("A cat sleeps on a"));
    ends.insert("scene", FieldEnd::model("kitchen").with_cut("n"));
    for raw in [
      "",
      "A cat sleeps on a",
      r#"{"description":"A cat sleeps on a","scene":"kitchen"}"#,
    ] {
      for ends in [&none, &ends] {
        assert_eq!(task.parse_ended(raw, ends), task.parse(raw), "{raw:?}");
      }
    }
  }

  /// LAW: **the provided `field_caps` declares no cap.**
  #[test]
  fn the_default_field_caps_declares_none() {
    let task = Measure {
      grammar: SmolStr::new("start: /.*/"),
    };
    assert!(task.field_caps().is_empty());
    assert_eq!(task.field_caps(), FieldCaps::default());
  }

  /// LAW: **`FieldCaps` maps a field's name to its one cap.** It starts
  /// empty (`new` is `default`); `insert` declares a cap and hands back the
  /// one it replaces; `get` reads a field by its exact name; `iter` walks the
  /// fields in name order.
  #[test]
  fn field_caps_maps_a_field_to_its_cap() {
    let mut caps = FieldCaps::new();
    assert_eq!(caps, FieldCaps::default());
    assert!(caps.is_empty());
    assert_eq!(caps.insert("description", 120), None);
    assert_eq!(caps.insert("scene", 0), None);
    assert_eq!(
      caps.insert("description", 80),
      Some(120),
      "the replaced cap"
    );
    assert_eq!(caps.len(), 2);
    assert_eq!(caps.get("description"), Some(80));
    assert_eq!(caps.get("Description"), None, "names are exact");
    let fields: Vec<(&str, usize)> = caps.iter().collect();
    assert_eq!(fields, [("description", 80), ("scene", 0)]);
    let copy = caps.clone();
    assert_eq!(copy, caps);
  }

  /// LAW: **`FieldEnds` maps a field's name to its one account.** It starts
  /// empty (`new` is `default`); `insert` records an account and hands back
  /// the one it replaces; `get` reads a field by its exact name; `iter`
  /// walks the fields in name order; two maps are equal when every field's
  /// account is.
  #[test]
  fn field_ends_maps_a_field_to_its_account() {
    let mut ends = FieldEnds::new();
    assert_eq!(ends, FieldEnds::default());
    assert!(ends.is_empty());
    assert_eq!(ends.len(), 0);
    assert!(ends.iter().next().is_none());

    let capped = FieldEnd::cap("A cat sleeps on a");
    let closed = FieldEnd::model("A cat sleeps.");
    assert_eq!(ends.insert("description", capped.clone()), None);
    assert_eq!(ends.insert("scene", FieldEnd::model("kitchen")), None);
    assert_eq!(
      ends.insert("description", closed.clone()),
      Some(capped),
      "the replaced account is handed back"
    );
    assert_eq!(ends.len(), 2);
    assert!(!ends.is_empty());
    assert_eq!(ends.get("description"), Some(&closed));
    assert_eq!(ends.get("Description"), None, "names are exact");
    assert_eq!(ends.get("tags"), None);
    let fields: Vec<(&str, bool)> = ends
      .iter()
      .map(|(field, end)| (field, end.closed_at_cap()))
      .collect();
    assert_eq!(fields, [("description", false), ("scene", false)]);

    let copy = ends.clone();
    assert_eq!(copy, ends);
    let mut other = copy;
    other.insert("scene", FieldEnd::cap("kitchen"));
    assert_ne!(other, ends, "one field's account differs");
  }
}
