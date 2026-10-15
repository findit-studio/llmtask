//! What the prompt-template setters allocate, observed through a counting
//! global allocator: a template they refuse is never copied, and one they
//! accept costs room bounded by `ImageAnalysisTask::PROMPT_MAX_BYTES`.
#![cfg(feature = "json")]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  cell::Cell,
};

use llmtask::{
  Task,
  image_analysis::{Extension, ImageAnalysisTask, InvalidTemplate},
};

/// The system allocator, recording the largest allocation the current
/// thread makes while it counts.
struct Counting;

thread_local! {
  // Neither needs a destructor, so reading them never allocates.
  static COUNTING: Cell<bool> = const { Cell::new(false) };
  static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// Records an allocation of `size` bytes if the current thread counts.
fn record(size: usize) {
  if COUNTING.try_with(Cell::get).unwrap_or(false) {
    let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(size)));
  }
}

// SAFETY: every method forwards its arguments to `System` unchanged, so
// `Counting` keeps `GlobalAlloc`'s contract exactly as `System` does, and
// `record` touches only thread-locals that never allocate.
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    record(layout.size());
    // SAFETY: forwarded unchanged; the caller keeps `alloc`'s contract.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    record(layout.size());
    // SAFETY: forwarded unchanged; the caller keeps `alloc_zeroed`'s contract.
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    record(new_size);
    // SAFETY: forwarded unchanged; the caller keeps `realloc`'s contract.
    unsafe { System.realloc(ptr, layout, new_size) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: forwarded unchanged; the caller keeps `dealloc`'s contract.
    unsafe { System.dealloc(ptr, layout) }
  }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f`, returning what it returns and the largest allocation, in bytes,
/// it made on this thread.
fn largest_allocation<T>(f: impl FnOnce() -> T) -> (T, usize) {
  LARGEST.with(|largest| largest.set(0));
  COUNTING.with(|counting| counting.set(true));
  let value = f();
  COUNTING.with(|counting| counting.set(false));
  (value, LARGEST.with(Cell::get))
}

/// The most a refusal may allocate: room for an unknown placeholder's name,
/// cut to 64 bytes, and its reference counts.
const SMALL: usize = 256;

/// LAW: a template refused for its widest rendering is never copied. A
/// million `{extensions}`, 12 MB rendering 551,000,000 bytes at its widest,
/// is refused at both setters, whatever the task's extensions are, with no
/// allocation beyond a small constant.
#[test]
fn a_template_refused_for_its_widest_rendering_is_never_copied() {
  let template = "{extensions}".repeat(1_000_000);
  let refusal = InvalidTemplate::TooLong {
    widest: 551_000_000,
    max: ImageAnalysisTask::PROMPT_MAX_BYTES,
  };
  for extensions in [&[][..], &Extension::ALL[..]] {
    let mut task = ImageAnalysisTask::new().with_extensions(extensions.iter().copied());
    let (set, largest) = largest_allocation(|| task.set_prompt_template(template.as_str()).err());
    assert_eq!(set, Some(refusal.clone()), "extensions {extensions:?}");
    assert!(
      largest <= SMALL,
      "set_prompt_template allocated {largest} bytes refusing it"
    );

    let task = task.clone();
    let (with, largest) = largest_allocation(|| task.with_prompt_template(template.as_str()).err());
    assert_eq!(with, Some(refusal.clone()), "extensions {extensions:?}");
    assert!(
      largest <= SMALL,
      "with_prompt_template allocated {largest} bytes refusing it"
    );
  }
}

/// LAW: a 1 MB unknown placeholder is refused with its name cut to 64 bytes,
/// and the refusal allocates no more than that name.
#[test]
fn a_huge_unknown_placeholder_is_refused_with_a_bounded_name() {
  let template = format!("{{{}}}", "n".repeat(1 << 20));
  let mut task = ImageAnalysisTask::new();
  let (refusal, largest) = largest_allocation(|| task.set_prompt_template(template.as_str()).err());
  assert_eq!(
    refusal,
    Some(InvalidTemplate::UnknownPlaceholder {
      name: "n".repeat(64).into(),
      cut: true,
    })
  );
  assert!(
    largest <= SMALL,
    "set_prompt_template allocated {largest} bytes refusing it"
  );
}

/// LAW: an accepted template costs room bounded by the prompt bound. A run
/// of 65,536 doubled braces, 128 KiB of source rendering 64 KiB, is stored
/// as its copied source, one 64 KiB literal run and the 64 KiB prompt; no
/// allocation passes twice the bound.
#[test]
fn an_accepted_template_costs_room_bounded_by_the_bound() {
  let max = ImageAnalysisTask::PROMPT_MAX_BYTES;
  let template = "{{".repeat(max);
  let mut task = ImageAnalysisTask::new();
  let (accepted, largest) =
    largest_allocation(|| task.set_prompt_template(template.as_str()).is_ok());
  assert!(accepted, "a template at the bound is accepted");
  assert!(
    largest <= 2 * max,
    "set_prompt_template allocated {largest} bytes for it"
  );
  assert_eq!(task.prompt(), "{".repeat(max));
}
