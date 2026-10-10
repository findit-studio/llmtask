//! `Grammar` — engine-agnostic constrained-decoding grammar.
//!
//! A `Task` produces a `Grammar` describing the allowed shape of
//! the model's output. Engines are free to implement whichever
//! variants they support natively:
//!
//! - JSON Schema is the lowest common denominator: every modern
//!   constrained-decoding engine (mistralrs, llguidance, OpenAI's
//!   structured output, vLLM) accepts it. Gated on the `json`
//!   feature (default-on).
//! - Lark / Regex are llguidance-native; engines built on
//!   llguidance (the `lfm` crate) consume them directly. Engines
//!   that only know JSON schema (mistralrs at present) reject
//!   them with [`UnsupportedGrammar`].
//!
//! New variants belong here, not in engine crates: that keeps the
//! enumeration of supported grammar surfaces in one place and lets
//! every consumer match exhaustively without scattering
//! engine-specific shimming.

use smol_str::SmolStr;

/// A constrained-decoding grammar produced by a [`crate::Task`].
///
/// Engines pattern-match on the variant and route to whichever
/// constraint mechanism they support. Use [`Grammar::is_json_schema`]
/// for the common "do I have JSON Schema?" check at API boundaries
/// that only accept JSON Schema (e.g., mistralrs's
/// `Constraint::JsonSchema`).
///
/// **Variant choice for `Task` implementors:** prefer JSON Schema
/// when your output is structured data (objects, arrays, typed
/// fields) — it works on every engine. Use Lark when you need
/// CFG-level constraints JSON Schema can't express (e.g.,
/// arithmetic expressions, indented blocks). Use Regex for simple
/// token-stream shapes (e.g., timestamp formats, single-token
/// enumerations).
///
/// `PartialEq`/`Eq` are NOT derived because [`Grammar::Regex`]
/// holds a `regex::Regex` (no Eq impl). Pattern-match on the
/// variant or compare the inner pattern string via
/// [`regex::Regex::as_str`] when comparison is needed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Grammar {
  /// JSON Schema (RFC 8927-style draft, as accepted by serde_json
  /// `Value`). The widest-compatible variant; all current engines
  /// accept it. Gated on the `json` feature.
  ///
  /// An engine that serves this crate's own JSON task,
  /// [`ImageAnalysisTask`](crate::image_analysis::ImageAnalysisTask),
  /// must honour every keyword its schema uses: `type`, `properties`,
  /// `required`, `additionalProperties: false`, `items`, `maxLength`
  /// (a string's length in Unicode scalar values) and `maxItems` (an
  /// array's element count). The schema uses no other: an engine refuses a
  /// schema with a keyword it does not implement. The task's `parse` holds
  /// the answer to the same keywords and to nothing looser, so it returns
  /// `Ok` only for an answer the schema accepts — or one its list rule
  /// brings to the schema, every such list marked `Capped`: a list over its
  /// `maxItems` is read to its cap, and an answer cut off inside a list's
  /// array is closed after the list's last whole item (see
  /// `ImageAnalysisTask`). An answer the schema accepts is
  /// a JSON text, with nothing but JSON whitespace around its value, in
  /// which every field has exactly the JSON type the schema declares for
  /// it. A string where the schema
  /// declares an array of strings (a comma-separated list included) and an
  /// array where it declares a string (a one-element array included) are
  /// refused like any other violation, by name, as
  /// [`JsonParseError::MissingFields`](crate::JsonParseError::MissingFields);
  /// an undeclared key is
  /// [`JsonParseError::UnknownFields`](crate::JsonParseError::UnknownFields).
  /// A decoder that enforces this schema never produces such a shape, so
  /// `parse` tolerates none, and an engine that ignores a keyword surfaces
  /// as a parse error — or, for a list's `maxItems`, as a list marked
  /// `Capped` — rather than as silently over-long or reshaped output.
  ///
  /// Only an answer that is not a JSON text is
  /// [`JsonParseError::Json`](crate::JsonParseError::Json): text serde_json's
  /// grammar rejects, or a string that escapes one half of a UTF-16
  /// surrogate pair (`\uD800` alone), which no Unicode text can hold,
  /// wherever that string sits — except an answer cut off inside a list's
  /// array, which is read as above. Every other answer reaches
  /// the checks above whole, however large or small its numbers and however
  /// deep its nesting: `1e400` where a string is declared is `MissingFields`
  /// naming the field, and under an undeclared key it is `UnknownFields`.
  #[cfg(feature = "json")]
  #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
  JsonSchema(serde_json::Value),
  /// Lark-format CFG grammar (llguidance's superset of EBNF).
  /// Native to llguidance-backed engines; engines without
  /// llguidance reject this variant.
  Lark(SmolStr),
  /// Regex pattern (RE2-style, anchor-implicit). Wraps a private
  /// [`RegexGrammar`] that holds the source pattern AND a compiled
  /// `regex::Regex` built from that exact pattern with default
  /// options — so [`Grammar::as_regex`] and
  /// [`Grammar::as_regex_pattern`] always describe the same
  /// language. Gated on the `regex` feature.
  ///
  /// **Engine parity caveat:** the Rust `regex` crate and
  /// llguidance's regex engine are both RE2-style and broadly
  /// compatible, but not bit-identical at the edges of supported
  /// syntax (some Unicode classes, lookaround flavors). A pattern
  /// that compiles in this variant is *very likely* but not
  /// guaranteed to compile in llguidance.
  #[cfg(feature = "regex")]
  #[cfg_attr(docsrs, doc(cfg(feature = "regex")))]
  Regex(RegexGrammar),
}

/// Grammar payload for [`Grammar::Regex`] — owns the source
/// pattern string AND its compiled `regex::Regex` (default options,
/// no builder flags). Both fields are private; the only path to
/// construct one is through [`Grammar::regex`], which guarantees
/// the source pattern (handed to engines) and the compiled regex
/// (used by callers for local validation) describe the same
/// language.
///
/// A `Grammar::Regex(regex::Regex)` shape (no wrapper) would let
/// callers smuggle in a `RegexBuilder::case_insensitive(true)`
/// regex whose `as_str()` returned the plain pattern but whose
/// `is_match` matched additional case-flipped strings — a silent
/// divergence between local validation and engine constraint.
/// Wrapping in a private type whose fields are only set via
/// `Regex::new` (no builder options) closes that path.
///
/// A second compiled regex is also stored for full-match validation
/// (see [`is_full_match`](Self::is_full_match)). It's built from
/// the parsed HIR via `Hir::concat([Look::Start, parsed_hir,
/// Look::End])` — a syntax-preserving anchored matcher. Two
/// simpler approaches don't work:
///
/// - Wrapping the source via `Regex::new(r"\A(?:{pattern})\z")` is
///   not syntax-preserving. A valid verbose-mode pattern like
///   `(?x)[0-9]+ # comment` compiles bare and then explodes
///   wrapped, because the trailing comment swallows the injected
///   `\z`.
/// - Using `find()` + span equality on the bare regex doesn't
///   work either: `regex::Regex` is leftmost-first, so for pattern
///   `a|ab` against input `ab`, `find` returns the shorter `0..1`
///   match for `a` and the span check fails — even though `ab` IS
///   in the language.
///
/// HIR-level anchoring puts the anchors in the regex grammar
/// itself (not literal substring decoration), so
/// `regex::Regex::is_match` on the anchored variant correctly
/// answers the full-language-membership question for any pattern
/// the regex crate accepts.
///
/// Available behind the `regex` feature.
#[cfg(feature = "regex")]
#[cfg_attr(docsrs, doc(cfg(feature = "regex")))]
#[derive(Debug, Clone)]
pub struct RegexGrammar {
  pattern: SmolStr,
  compiled: regex::Regex,
  // HIR-anchored variant for `is_full_match`. Built from
  // `Hir::concat([Look::Start, parsed_hir, Look::End])`, then
  // re-emitted to a regex string via `Hir::to_string()` and
  // re-compiled. Re-compiling is cheap and unavoidable because
  // `regex::Regex` doesn't expose a "compile from HIR" entry
  // point on its public API.
  anchored: regex::Regex,
}

#[cfg(feature = "regex")]
#[cfg_attr(docsrs, doc(cfg(feature = "regex")))]
const _: () = {
  impl RegexGrammar {
    /// Borrow the source pattern string (unanchored, as supplied).
    /// This is what engines like llguidance receive
    /// (`TopLevelGrammar::from_regex(pattern)`) — they treat the
    /// pattern as anchor-implicit / full-match against the generated
    /// output.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn pattern(&self) -> &str {
      &self.pattern
    }

    /// Borrow the compiled `regex::Regex`. Guaranteed to be
    /// `Regex::new(self.pattern())` (default options, no builder
    /// flags), so it describes the same language as the pattern
    /// engines receive.
    ///
    /// **Substring vs full-match:** `Regex::is_match` is unanchored
    /// — it returns `true` for any substring match. Engines like
    /// llguidance treat the pattern as anchor-implicit / full-match.
    /// For engine-parity validation, prefer
    /// [`is_full_match`](Self::is_full_match), which uses an HIR-
    /// anchored compiled regex and gives true full-language
    /// membership semantics for arbitrary patterns.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn compiled(&self) -> &regex::Regex {
      &self.compiled
    }

    /// Engine-parity full-match validator: returns `true` iff the
    /// **whole** input is in the language of the pattern. Backed by
    /// a separate compiled regex that wraps the parsed HIR with
    /// `Look::Start`/`Look::End` anchors, so:
    ///
    /// - `(?x)[0-9]+ # comment` validates `"123"` — anchoring at
    ///   the HIR level can't be eaten by lexical verbose-mode
    ///   comments.
    /// - `a|ab` validates `"ab"` — with the anchors in the regex
    ///   grammar itself, leftmost-first correctly backtracks past
    ///   the shorter alternative.
    /// - `[0-9]{4}-[0-9]{2}-[0-9]{2}` rejects `"abc2026-05-09xyz"`
    ///   — substring matches are rejected.
    ///
    /// Use this in `Task::parse` to defend against an engine that
    /// ignores or under-applies the regex constraint.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn is_full_match(&self, input: &str) -> bool {
      self.anchored.is_match(input)
    }
  }

  impl Grammar {
    /// Sole constructor for [`Grammar::Regex`]: compiles the pattern
    /// with `regex::Regex::new` (default options, no builder flags)
    /// and stores both the source pattern string and the compiled
    /// regex inside a [`RegexGrammar`]. The two are guaranteed to
    /// describe the same language because they're built from the
    /// same input with no additional options.
    ///
    /// **Why no `from_compiled_regex` constructor:** allowing a
    /// `regex::Regex` built via `RegexBuilder::case_insensitive(true)`
    /// (or any other builder flag) would let `as_regex()` validate
    /// one language while `as_regex_pattern()` exported a different
    /// one to the engine. Forcing construction through this
    /// string-based path eliminates the divergence.
    ///
    /// **Engine parity caveat:** see the note on [`Grammar::Regex`].
    /// A pattern that compiles here is *very likely* but not
    /// guaranteed to compile in llguidance.
    pub fn regex(pattern: &str) -> Result<Self, regex::Error> {
      // Compile the bare pattern first — surfaces a `regex::Error`
      // referencing the user's literal input. `regex` and
      // `regex-syntax` share the same parser internally, so any
      // pattern that compiles here also parses to a valid HIR.
      let compiled = regex::Regex::new(pattern)?;
      // Build a syntax-preserving anchored validator via HIR.
      // `Hir::concat([Look::Start, parsed_hir, Look::End])` puts
      // the anchors in the regex grammar itself, so:
      //   - leftmost-first correctly backtracks past short prefixes
      //     in alternations (`a|ab` matches `ab`, not just `a`);
      //   - verbose-mode lexical features (`(?x)…# comment`) can't
      //     swallow the anchors the way raw string interpolation
      //     would.
      let parsed_hir = regex_syntax::Parser::new()
        .parse(pattern)
        .expect("regex::Regex::new accepted this pattern, regex-syntax must too");
      let anchored_hir = regex_syntax::hir::Hir::concat(std::vec![
        regex_syntax::hir::Hir::look(regex_syntax::hir::Look::Start),
        parsed_hir,
        regex_syntax::hir::Hir::look(regex_syntax::hir::Look::End),
      ]);
      // `std::format!` resolves to `alloc::format!` under no_std
      // thanks to the `extern crate alloc as std;` alias in lib.rs;
      // it gives us a `String` from `Hir`'s `Display` impl without
      // needing `ToString` in scope.
      let anchored = regex::Regex::new(&std::format!("{anchored_hir}"))
        .expect("HIR-emitted anchored variant of valid pattern must compile");
      Ok(Self::Regex(RegexGrammar {
        pattern: SmolStr::new(pattern),
        compiled,
        anchored,
      }))
    }

    /// Borrow the compiled `regex::Regex` for the [`Grammar::Regex`]
    /// variant. Returns `None` for non-regex variants.
    ///
    /// **Substring vs full-match — use [`is_regex_full_match`]:**
    /// the returned regex is `Regex::new(pattern)` with default
    /// options, so `is_match` is unanchored and would accept
    /// substrings the engine grammar (anchor-implicit) wouldn't have
    /// produced. For engine-parity validation in `Task::parse`, use
    /// [`Grammar::is_regex_full_match`] (or
    /// [`RegexGrammar::is_full_match`] directly) — it goes through
    /// an HIR-anchored matcher and gives full-language-membership
    /// semantics for arbitrary regex syntax (including verbose-mode
    /// patterns that break naive `\A(?:p)\z` string-anchoring).
    ///
    /// Available behind the `regex` feature.
    ///
    /// [`is_regex_full_match`]: Self::is_regex_full_match
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn as_regex(&self) -> Option<&regex::Regex> {
      if let Self::Regex(rg) = self {
        Some(rg.compiled())
      } else {
        None
      }
    }

    /// Borrow the regex pattern as a `&str`, for engines that need
    /// the raw pattern (e.g., llguidance's
    /// `TopLevelGrammar::from_regex`). Returns `None` for non-regex
    /// variants. Convenience for `as_regex().map(|r| r.as_str())`.
    /// Available behind the `regex` feature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn as_regex_pattern(&self) -> Option<&str> {
      if let Self::Regex(rg) = self {
        Some(rg.pattern())
      } else {
        None
      }
    }

    /// Engine-parity full-match validator for the [`Grammar::Regex`]
    /// variant: returns `Some(true)` iff the compiled regex matches
    /// the entire `input`, `Some(false)` if it matches only a
    /// substring (or doesn't match at all), and `None` for non-regex
    /// variants. Thin wrapper over [`RegexGrammar::is_full_match`].
    /// Use this in `Task::parse` to defend against an engine that
    /// ignores or under-applies the regex constraint. Available
    /// behind the `regex` feature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn is_regex_full_match(&self, input: &str) -> Option<bool> {
      if let Self::Regex(rg) = self {
        Some(rg.is_full_match(input))
      } else {
        None
      }
    }
  }
};

#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
const _: () = {
  impl Grammar {
    /// Convenience constructor for [`Grammar::JsonSchema`]. Available
    /// behind the `json` feature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn json_schema(value: serde_json::Value) -> Self {
      Self::JsonSchema(value)
    }

    /// Returns `Some(&Value)` when this grammar is JSON Schema, else
    /// `None`. Engines that only accept JSON Schema use this to
    /// route or reject. Available behind the `json` feature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn as_json_schema(&self) -> Option<&serde_json::Value> {
      if let Self::JsonSchema(v) = self {
        Some(v)
      } else {
        None
      }
    }

    /// `true` iff this grammar is the JSON Schema variant. Available
    /// behind the `json` feature.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn is_json_schema(&self) -> bool {
      matches!(self, Self::JsonSchema(_))
    }
  }
};

impl Grammar {
  /// Convenience constructor for [`Grammar::Lark`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn lark(src: impl Into<SmolStr>) -> Self {
    Self::Lark(src.into())
  }

  /// Short, stable name for the variant — for diagnostics and
  /// the [`UnsupportedGrammar`] error. Returns `"json_schema"` /
  /// `"lark"` / `"regex"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kind(&self) -> &'static str {
    match self {
      #[cfg(feature = "json")]
      Self::JsonSchema(_) => "json_schema",
      Self::Lark(_) => "lark",
      #[cfg(feature = "regex")]
      Self::Regex(_) => "regex",
    }
  }
}

/// Returned by an engine when a [`Task`](crate::Task) supplies a
/// [`Grammar`] variant the engine doesn't support. Carries the
/// rejected variant's [`Grammar::kind`] so callers can route to
/// a different engine.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[error("engine does not support `{kind}` grammar (supported: {supported})")]
pub struct UnsupportedGrammar {
  /// The [`Grammar::kind`] string of the rejected variant.
  pub kind: &'static str,
  /// Human-readable list of variants the engine does support.
  pub supported: &'static str,
}

impl UnsupportedGrammar {
  /// Construct an `UnsupportedGrammar` for the given rejected
  /// variant kind.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(kind: &'static str, supported: &'static str) -> Self {
    Self { kind, supported }
  }
}

// Tests run under both std (default) and `--no-default-features
// --features alloc`: `to_string()` resolves through
// `alloc::string::ToString`, which we bring in explicitly so the
// std prelude isn't required.
#[cfg(test)]
mod tests {
  use super::*;
  use std::string::ToString;

  #[cfg(feature = "json")]
  use serde_json::json;

  #[cfg(feature = "json")]
  #[test]
  fn grammar_kind_strings_are_stable_json() {
    assert_eq!(Grammar::json_schema(json!({})).kind(), "json_schema");
  }

  #[test]
  fn grammar_kind_strings_are_stable_lark() {
    assert_eq!(Grammar::lark("start: \"a\"").kind(), "lark");
  }

  #[cfg(feature = "regex")]
  #[test]
  fn grammar_kind_strings_are_stable_regex() {
    assert_eq!(Grammar::regex(r"[0-9]+").unwrap().kind(), "regex");
  }

  #[cfg(feature = "json")]
  #[test]
  fn as_json_schema_only_returns_json_variant() {
    let js = Grammar::json_schema(json!({"type":"string"}));
    assert!(js.as_json_schema().is_some());
    assert!(js.is_json_schema());

    let lark = Grammar::lark("start: \"x\"");
    assert!(lark.as_json_schema().is_none());
    assert!(!lark.is_json_schema());
  }

  #[test]
  fn unsupported_grammar_message_includes_both_kinds() {
    let err = UnsupportedGrammar::new("lark", "json_schema");
    let msg = err.to_string();
    assert!(msg.contains("lark"));
    assert!(msg.contains("json_schema"));
  }

  #[cfg(feature = "json")]
  #[test]
  fn grammar_is_clone() {
    // Eq dropped (Regex isn't Eq); just verify Clone works and
    // produces a structurally-similar value.
    let a = Grammar::json_schema(json!({"type":"object"}));
    let b = a.clone();
    assert_eq!(a.kind(), b.kind());
    assert_eq!(a.as_json_schema(), b.as_json_schema());
  }

  // ===== `regex` feature =====

  #[cfg(feature = "regex")]
  #[test]
  fn regex_constructor_accepts_valid_pattern() {
    let g = Grammar::regex(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$").expect("valid pattern");
    assert!(matches!(g, Grammar::Regex(_)));
    assert_eq!(g.kind(), "regex");
  }

  #[cfg(feature = "regex")]
  #[test]
  fn regex_constructor_rejects_invalid_pattern() {
    let result = Grammar::regex(r"[a-z");
    assert!(result.is_err(), "invalid pattern must reject");
  }

  #[cfg(feature = "regex")]
  #[test]
  fn regex_pattern_and_compiled_describe_same_language() {
    // The only constructor is `Grammar::regex(&str)`, which builds
    // `compiled` via `Regex::new` with no builder options. A
    // `from_compiled_regex(re)` would have let callers smuggle in
    // a `RegexBuilder::case_insensitive(true)` regex whose
    // `as_str()` returned the bare pattern but whose `is_match`
    // matched additional case-flipped strings. Invariants:
    //   Grammar::regex(p).as_regex_pattern() == Some(p)
    //   Grammar::regex(p).as_regex().unwrap().as_str() == p
    //   case-flipped input does NOT match (no case_insensitive)
    let g = Grammar::regex(r"yes|no").unwrap();
    assert_eq!(g.as_regex_pattern(), Some("yes|no"));
    let re = g.as_regex().expect("Some for Regex variant");
    assert_eq!(re.as_str(), "yes|no");
    assert!(re.is_match("yes"));
    assert!(re.is_match("no"));
    assert!(
      !re.is_match("YES"),
      "case-insensitive flag MUST NOT be smuggleable"
    );
    assert!(!re.is_match("NO"));
  }

  #[cfg(feature = "regex")]
  #[test]
  fn as_regex_returns_borrow_for_regex_variant() {
    let g = Grammar::regex(r"[0-9]+").unwrap();
    let r = g.as_regex().expect("Some for Regex variant");
    assert!(r.is_match("42"));
    assert!(!r.is_match("abc"));
  }

  #[cfg(feature = "regex")]
  #[test]
  fn as_regex_pattern_returns_str_for_regex_variant() {
    let g = Grammar::regex(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
    assert_eq!(g.as_regex_pattern(), Some(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"));
  }

  #[cfg(feature = "regex")]
  #[test]
  fn is_full_match_rejects_substring_padded_match() {
    // `as_regex().unwrap().is_match(raw)` is unanchored (substring),
    // so callers using it directly for engine-parity validation
    // would accept outputs llguidance never would have produced.
    // `is_full_match` is the engine-parity validator. This test
    // pins its substring-rejection behavior.
    let g = Grammar::regex(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").unwrap();
    assert_eq!(g.is_regex_full_match("2026-05-09"), Some(true));
    assert_eq!(
      g.is_regex_full_match("abc2026-05-09xyz"),
      Some(false),
      "leading + trailing junk"
    );
    assert_eq!(
      g.is_regex_full_match("2026-05-09trailing"),
      Some(false),
      "trailing junk"
    );
    assert_eq!(
      g.is_regex_full_match("leading2026-05-09"),
      Some(false),
      "leading junk"
    );
    assert_eq!(g.is_regex_full_match("not a date"), Some(false));
    // Alternation-bounding case: the unanchored `is_match` on
    // `yes|no` would accept "yesno"; full-match correctly rejects.
    let g = Grammar::regex(r"yes|no").unwrap();
    assert_eq!(g.is_regex_full_match("yes"), Some(true));
    assert_eq!(g.is_regex_full_match("no"), Some(true));
    assert_eq!(g.is_regex_full_match("yesno"), Some(false));
  }

  #[cfg(feature = "regex")]
  #[test]
  fn is_full_match_handles_verbose_mode_patterns() {
    // Naive anchoring via `Regex::new(format!(r"\A(?:{p})\z"))`
    // would break here: in verbose mode, the trailing `# comment`
    // swallows the injected `\z` to end-of-line, so construction
    // fails for valid input. HIR-level anchoring is immune — the
    // parser consumes the comment before anchors are added.
    let g = Grammar::regex(r"(?x)[0-9]+ # trailing comment").unwrap();
    assert_eq!(g.is_regex_full_match("123"), Some(true));
    assert_eq!(g.is_regex_full_match("abc123xyz"), Some(false));
    assert_eq!(g.is_regex_full_match("123 trailing"), Some(false));
  }

  #[cfg(feature = "regex")]
  #[test]
  fn is_full_match_handles_prefix_alternatives() {
    // A `find()` + span-equality implementation would fail here:
    // `regex::Regex` is leftmost-first, so for `a|ab` against
    // input `ab` it returns the shorter `0..1` match for `a`,
    // making the span check think `ab` isn't in the language.
    // With HIR-anchored compilation the anchors are inside the
    // regex grammar, so leftmost-first correctly backtracks past
    // the short alternative when it can't satisfy the trailing
    // `\z`.
    let g = Grammar::regex(r"a|ab").unwrap();
    assert_eq!(g.is_regex_full_match("a"), Some(true), "short arm");
    assert_eq!(g.is_regex_full_match("ab"), Some(true), "long arm");
    assert_eq!(g.is_regex_full_match("abc"), Some(false));

    // Empty alternative — `|a` should match `a` (the `a` arm) AND
    // the empty string (the empty arm).
    let g = Grammar::regex(r"|a").unwrap();
    assert_eq!(g.is_regex_full_match(""), Some(true), "empty arm");
    assert_eq!(g.is_regex_full_match("a"), Some(true), "non-empty arm");
    assert_eq!(g.is_regex_full_match("aa"), Some(false));

    // Sanity: nested alternation with prefix overlap.
    let g = Grammar::regex(r"foo|foobar|foob").unwrap();
    assert_eq!(g.is_regex_full_match("foo"), Some(true));
    assert_eq!(g.is_regex_full_match("foob"), Some(true));
    assert_eq!(g.is_regex_full_match("foobar"), Some(true));
    assert_eq!(g.is_regex_full_match("foobaz"), Some(false));
  }

  #[cfg(all(feature = "regex", feature = "json"))]
  #[test]
  fn is_regex_full_match_returns_none_for_non_regex_variants() {
    assert_eq!(
      Grammar::json_schema(json!({})).is_regex_full_match("anything"),
      None
    );
    assert_eq!(
      Grammar::lark("start: \"x\"").is_regex_full_match("anything"),
      None
    );
  }

  #[cfg(all(feature = "regex", feature = "json"))]
  #[test]
  fn as_regex_returns_none_for_non_regex_variant() {
    assert!(Grammar::json_schema(json!({})).as_regex().is_none());
    assert!(Grammar::lark("start: \"x\"").as_regex().is_none());
    assert!(Grammar::lark("start: \"x\"").as_regex_pattern().is_none());
  }
}
