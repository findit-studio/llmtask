//! `Task` trait and parse-error types — the cross-engine
//! abstraction.
//!
//! `Task::ParseError` is an associated type so the trait does not
//! bake in any concrete error representation. Tasks that parse
//! JSON output use [`crate::JsonParseError`] (gated on the `json`
//! feature); Tasks that parse free-form text or custom formats
//! choose their own error type.

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
/// Implementors provide all four method signatures explicitly —
/// there are no default implementations. The trait used to expose
/// only `schema(&self) -> &serde_json::Value` and a fixed
/// `ParseError` enum, but both have been generalized:
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
    /// `serde_json` failed to parse the response as valid JSON.
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
    /// decode their top-level object through a duplicate-checking parse
    /// (e.g. `image_analysis::ImageAnalysisTask::parse`) return this
    /// instead, naming the repeated key, before a `Value` is ever built.
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
    /// holding anything but strings. All four cases are folded into one
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
