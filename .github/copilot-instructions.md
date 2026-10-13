# AI architecture rubric for databench

## When proposing or making changes, optimize for lower cognitive load, explicit data flow, and fewer moving parts.

Prefer concrete, explicit, single-purpose code. Do not add abstraction, flexibility, or automation unless the current code already demonstrates stable repeated need.

1. Preserve the core model
	•	databench is a thin procedural toolkit over pandas, not a framework.
	•	Bench is a small coordinator, not a smart orchestration engine.
	•	The default workflow stays explicit: load -> filter -> analyze -> plot -> save.

2. Protect the data contract
	•	Treat table shapes and column names as hard contracts, not flexible hints.
	•	Prefer adapting new datasets in scripts to the canonical schema.
	•	Do not add schema inference, alias resolution, fallback column guessing, or input normalization layers unless explicitly requested.

3. Keep behavior explicit
	•	Prefer explicit imports, explicit column names, and explicit pipeline steps.
	•	Do not add hidden behavior, convenience automation, or implicit workflow steps.
	•	Fail clearly on missing required inputs instead of silently inferring intent.

4. Resist abstraction

Before introducing any helper, class, or API layer, ask:
	•	Does this remove real duplication already present in multiple places?
	•	Does this reduce the number of concepts a reader must hold in mind?
	•	Would direct procedural code be easier to read and edit?

If the answer is not clearly yes, do not abstract.

5. Keep Bench thin

Do not turn Bench into a place for:
	•	plugin discovery
	•	schema inference
	•	config-driven execution
	•	factory dispatch
	•	workflow automation
	•	dataset-specific branching

Bench should coordinate explicit steps, not hide them.

6. Keep dataset-specific logic in scripts
	•	ROI mapping, condition labeling, and one-off dataset reshaping belong in scripts unless the same pattern is repeated and stable across multiple scripts.
	•	Do not promote one dataset’s quirks into shared library abstractions prematurely.

7. Do not “future-proof”

Avoid adding:
	•	generic base classes
	•	strategy hierarchies
	•	factory layers
	•	plugin managers
	•	auto-discovery
	•	config objects
	•	multi-mode APIs
	•	extensibility hooks for hypothetical use cases

Prefer one concrete implementation for the current need.

8. Prefer local reasoning

A good change makes it easier to understand a feature by reading a small amount of code nearby.
Prefer:
	•	narrow interfaces
	•	direct data transformations
	•	concrete variable names
	•	simple pandas/numpy operations
	•	short, readable scripts

Avoid:
	•	cross-cutting indirection
	•	state hidden across layers
	•	wrappers that add little value
	•	abstractions that require jumping across many files to understand behavior

9. Use descriptive names — code is the documentation

Variable, function, and parameter names must be self-documenting. A reader should understand what a value represents without tracing its origin or checking comments.

Naming rules:
	•	Use full, descriptive names: `filtered_signal`, `envelope`, `threshold_value`, `prev_start`, `prev_end` — not `xf`, `env`, `thr`, `ps`, `pe`.
	•	Loop variables and array intermediates get real names: `region_key`, `burst_start`, `time_valid` — not `k`, `s`, `tt`.
	•	Dataclass fields, function parameters, and return values are part of the public contract — name them as if they were documentation headings.
	•	Numpy/pandas temporaries inside a function still get readable names: `bandpassed`, `analytic`, `detrended` — not `xb`, `hb`, `xd`.
	•	One-letter names are acceptable only for trivial, immediately obvious scope: `i` in a short `enumerate` loop, `x` and `y` in a two-line lambda.

The codebase relies on naming rather than comments for inline documentation. If a name needs a comment to explain it, the name is wrong.

10. Accept duplication when it is cheaper than indirection

A small amount of repeated, obvious code is often better than a shared abstraction with extra parameters, modes, or hidden assumptions.

11. Review checklist for every change

Reject or simplify the change if it:
	•	adds a new abstraction without repeated need
	•	makes the workflow less explicit
	•	broadens the API for hypothetical reuse
	•	introduces multiple ways to do the same thing
	•	hides data reshaping or analysis steps
	•	teaches the library to support many input variants instead of normalizing in scripts
	•	increases the number of concepts needed to trace a pipeline

Prefer the change if it:
	•	removes indirection
	•	collapses layers
	•	keeps behavior visible in scripts
	•	preserves a fixed schema
	•	shortens the path from input table to result
	•	makes the code easier for a human to edit quickly

Default tie-breaker

When two designs both work, choose the one with:
	1.	fewer concepts
	2.	fewer extension points
	3.	less hidden behavior
	4.	more explicit data flow
	5.	better local readability
