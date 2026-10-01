<div align="center">
<h1>llmtask</h1>
</div>
<div align="center">

Engine-agnostic structured-output abstraction for LLMs — `Task` trait, `Grammar` enum (JSON Schema, Lark, Regex), and the canonical `ImageAnalysis` data type. Decouples a prompt + grammar + parser from any specific inference backend.

[<img alt="github" src="https://img.shields.io/badge/github-findit--studio/llmtask-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fllmtask" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-studio/llmtask/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-studio/llmtask?style=for-the-badge&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-llmtask-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/llmtask?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/llmtask?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCAxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTYgYy0xMi4xMTgtMTcuOTM3LTMyLjI2Mi0yOC42NDUtNTMuODgyLTI4LjY0NSBjLTM1LjgzMywwLTY0Ljk4NSwyOS4xNTItNjQuOTg1LDY0Ljk4NnMyOS4xNTIsNjQuOTg2LDY0Ljk4NSw2NC45ODZjMjIuMjgxLDAsNDIuNzU5LTExLjIxOCw1NC43NzgtMzAuMDA5IEMyMDAuMjA4LDE2OS4xNDcsMTk4Ljk4NSwxNjMuNTgyLDE5NC43OTgsMTYwLjkwM3oiLz4gPC9nPiA8L2c+Cg08L3N2Zz4=" height="22">

</div>

## Overview

`llmtask` is the engine-agnostic side of every "structured output from an LLM" pipeline:

- **[`Task`]** — a trait carrying the four things every constrained-decoding call needs: a prompt, a borrowed schema (`type Value`), a grammar wrapper (`Grammar` enum), and a typed parser (`type Output`, `type ParseError`). Engines accept any `T: Task<Value = ...>`, so a `Task` written once runs against any engine in the ecosystem (`lfm`, `qwen`, …) without translation.
- **[`Grammar`]** — an enum over the constrained-decoding surfaces real engines accept: JSON Schema (`Grammar::JsonSchema`, behind the `json` feature), Lark (`Grammar::Lark`), and Regex (`Grammar::Regex(RegexGrammar)`, behind the `regex` feature — the wrapper holds both the source pattern and a default-options compiled regex, guaranteeing engine grammar and local validation describe the same language). Engines pattern-match and return [`UnsupportedGrammar`] when they don't speak a given variant — the caller can then route to a different backend.
- **[`ImageAnalysis`]** — the canonical single-image VLM output shape (scene category, description, subjects/objects/actions/emotion/lighting lists, shot-type label, search tags, category labels). Lets multiple VLM engines (`lfm`, `qwen`) produce values of the same type so downstream consumers compare and merge results without per-engine adapters.
- **[`ImageAnalysisTask`]** — the canonical `Task` implementation producing an `ImageAnalysis`: prompt, JSON Schema (`additionalProperties: false`), and a parser that holds the answer to that schema, all three derived from one field roster. A field that is missing or null, of any JSON type other than the declared one, over a declared cap, or outside the schema is refused as a named `JsonParseError`, never coerced, truncated or silently accepted. Only an answer that is not JSON at all is `JsonParseError::Json`: a number of any magnitude and a value nested to any depth are wrong types like any other, refused by the field's name. The drift tolerances of the 0.3 parser are gone: a comma-separated string for `tags`, a bare string for a label array and a one-element array for `shot_type` are refused like any other wrong type, since a decoder that enforces the schema never produces them. By default the task asks for one length-capped `description` sentence and one count-capped `tags` array; each of the other eight fields is an `Extension` a deployment switches on with `with_extensions`. Behind the `json` feature. `lfm` and `qwen3-vl` both ran their own byte-for-byte-equivalent copy of this task before it moved up here.

[`Task`]: https://docs.rs/llmtask/latest/llmtask/task/trait.Task.html
[`Grammar`]: https://docs.rs/llmtask/latest/llmtask/grammar/enum.Grammar.html
[`UnsupportedGrammar`]: https://docs.rs/llmtask/latest/llmtask/grammar/struct.UnsupportedGrammar.html
[`ImageAnalysis`]: https://docs.rs/llmtask/latest/llmtask/image_analysis/struct.ImageAnalysis.html
[`ImageAnalysisTask`]: https://docs.rs/llmtask/latest/llmtask/image_analysis/struct.ImageAnalysisTask.html

## Why an engine-agnostic Task layer?

Without a shared trait, every "structured output" prompt re-implements the same plumbing per engine: prompt string, schema construction, parser, error type, validation predicate. The plumbing is engine-independent — the same `ImageAnalysis` JSON Schema works against `lfm`'s llguidance backend and `qwen`'s mistralrs backend; only the *constraint API* differs.

`llmtask` separates them:

```text
                                ┌──────────────────────────┐
   YourTask: impl Task   ──▶    │  llmtask::Task contract  │   ──▶  any engine that
                                │   prompt + Grammar       │       takes &impl Task
                                │   parse → Output         │
                                └──────────────────────────┘
```

A `Task` written today against a JSON Schema runs through `lfm` (llguidance) and `qwen` (mistralrs) unchanged. A `Task` returning a Lark grammar runs through `lfm` natively; `qwen` rejects it cleanly via `Error::UnsupportedGrammar` so the caller can dispatch elsewhere.

## Features

- **Engine-agnostic `Task` trait** with associated `Output`, `Value`, and `ParseError` types — engines bound to a specific schema kind get typed access (`fn run<T: Task<Value = serde_json::Value>>`).
- **Three grammar surfaces** in a single `#[non_exhaustive]` enum: JSON Schema (default), Lark, and pre-compiled `regex::Regex`. Engines pattern-match and route.
- **`UnsupportedGrammar` error** carrying the rejected variant `kind()` and the engine's `supported` list — callers can route to a different engine when one variant isn't accepted.
- **Optional `json` feature** (default-on) — `Grammar::JsonSchema(serde_json::Value)` plus the `JsonParseError` convenience type. Drop it via `default-features = false, features = ["alloc"]` (or `"regex"` / `"serde"`, both of which imply `alloc`) to get a Lark-or-Regex-only build with no `serde_json` dep. NOTE: `alloc` is required to reach any public API — `default-features = false` alone exposes nothing.
- **Optional `regex` feature** — pre-compiled `regex::Regex` in the variant (validation enforced by the type), plus `as_regex()` / `as_regex_pattern()` helpers.
- **Optional `serde` feature** — `Serialize` / `Deserialize` on `ImageAnalysis` for downstream wire formats.
- **Canonical `ImageAnalysis`** — ten-field single-image VLM output shape with builder-style API (`with_*` / `set_*`), shared across the findit-studio engines.
- **Canonical `ImageAnalysisTask`** (behind `json`) — the prompt + schema + parser that produces an `ImageAnalysis`, so every engine runs the same task instead of a per-engine copy.

## Example

```rust,ignore
// Marked `ignore` so doctests don't pull serde_derive under
// default features (`std + json` only). The example needs
// `--features serde` to compile (the `serde::Deserialize`
// derive); enable that on the dev-deps in any consumer who
// wants to lift this verbatim.
use std::sync::OnceLock;
use llmtask::{Grammar, ImageAnalysis, JsonParseError, Task};
use serde_json::{Value, json};

/// A minimal Task: "summarize what's in this image" as a JSON object
/// with a single `summary` string field.
struct SummaryTask;

impl Task for SummaryTask {
  type Output = String;
  type Value = Value;
  type ParseError = JsonParseError;

  fn prompt(&self) -> &str {
    "Reply with JSON: {\"summary\": \"<one sentence>\"}"
  }

  fn schema(&self) -> &Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| json!({
      "type": "object",
      "properties": { "summary": { "type": "string" } },
      "required": ["summary"],
    }))
  }

  fn grammar(&self) -> Grammar {
    Grammar::JsonSchema(self.schema().clone())
  }

  fn parse(&self, raw: &str) -> Result<String, JsonParseError> {
    #[derive(serde::Deserialize)]
    struct R { summary: String }
    let r: R = serde_json::from_str(raw.trim())?;
    Ok(r.summary)
  }
}

// `&SummaryTask` now satisfies any engine taking `&impl Task<Value = Value>`:
//   lfm::Engine::run(&SummaryTask, &images, &opts)
//   qwen::Engine::run(&SummaryTask, images).await

// And the canonical multi-field VLM output type lives right here:
let _empty = ImageAnalysis::new();
```

A regex-only Task (no JSON dep at all):

```rust,ignore
// Marked `ignore` so doctests don't try to compile this under
// default features (the `regex` feature is opt-in).
use llmtask::{Grammar, Task};
use smol_str::SmolStr;

struct TimestampTask {
  // Source pattern as the canonical Value — what engines like
  // llguidance receive (anchor-implicit / full-match semantics).
  pattern: SmolStr,
  // Cached `Grammar` so `parse` can call `is_regex_full_match`
  // without rebuilding the grammar each call.
  grammar: Grammar,
}

#[derive(Debug)]
struct StringErr(String);
impl std::fmt::Display for StringErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.0)
  }
}
impl std::error::Error for StringErr {}

impl Task for TimestampTask {
  type Output = String;
  type Value = SmolStr;
  type ParseError = StringErr;
  fn prompt(&self) -> &str { "Output a date in YYYY-MM-DD format." }
  fn schema(&self) -> &SmolStr { &self.pattern }
  fn grammar(&self) -> Grammar { self.grammar.clone() }
  fn parse(&self, raw: &str) -> Result<String, StringErr> {
    let trimmed = raw.trim();
    // `is_regex_full_match` is the engine-parity validator:
    // `find()` + span equality, so it agrees with llguidance's
    // anchor-implicit grammar. Bare `as_regex().is_match(...)` is
    // unanchored substring matching and would accept e.g.
    // `"abc2026-05-09xyz"` for `[0-9]{4}-[0-9]{2}-[0-9]{2}` — not
    // what the engine produced, so don't use it for validation.
    if self.grammar.is_regex_full_match(trimmed) != Some(true) {
      return Err(StringErr(format!("output {trimmed:?} does not match")));
    }
    Ok(trimmed.to_string())
  }
}
```

## Installation

```toml
[dependencies]
# Default: JSON Schema support on, Lark always available, regex off, serde off.
llmtask = "0.4"

# Lark-only build (no serde_json, no regex):
# `alloc` is required — without it the public API is empty.
llmtask = { version = "0.3", default-features = false, features = ["alloc"] }

# Regex-only build (no serde_json; `regex` already implies `alloc`):
llmtask = { version = "0.3", default-features = false, features = ["regex"] }

# Everything:
llmtask = { version = "0.3", features = ["json", "regex", "serde"] }
```

| Feature  | Default | What it adds                                                                                                                       |
| -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `json`   | yes     | `Grammar::JsonSchema(serde_json::Value)` variant + `JsonParseError` + the `serde_json` dep (with its additive `raw_value` feature) |
| `regex`  | no      | `Grammar::Regex(RegexGrammar)` variant + validating `Grammar::regex` constructor + `regex` dep                                     |
| `serde`  | no      | `Serialize` / `Deserialize` on `ImageAnalysis`                                                                                     |

## MSRV

Rust 1.95.

## License

`llmtask` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT Studio authors.

[Github-url]: https://github.com/findit-studio/llmtask/
[CI-url]: https://github.com/findit-studio/llmtask/actions/workflows/ci.yml
[doc-url]: https://docs.rs/llmtask
[crates-url]: https://crates.io/crates/llmtask
[codecov-url]: https://app.codecov.io/gh/findit-studio/llmtask/
