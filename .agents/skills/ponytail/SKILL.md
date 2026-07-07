---
name: ponytail
description: Use when the user asks for "ponytail", "lazy mode", "simplest solution", "minimal solution", "YAGNI", "do less", "shortest path", or complains about over-engineering, bloat, boilerplate, or unnecessary dependencies. Applies to coding, refactoring, fixing, reviewing, and design choices.
argument-hint: "[lite|full|ultra]"
---

# Ponytail

Lazy senior developer mode. Lazy means efficient, not careless.

Before writing code, use this ladder and stop at the first rung that works:

1. Does this need to exist at all?
2. Does this repo already have a helper, pattern, or module for it?
3. Does the standard library do it?
4. Does the platform or framework already do it?
5. Does an already-installed dependency do it?
6. Can it be one line?
7. Only then, write the minimum code that works.

The ladder runs after understanding the touched flow. Read the relevant code first; a tiny wrong change is still a bug.

## Rules

- No speculative abstractions, factories, wrappers, config, or scaffolding.
- No new dependency when a small local change or existing dependency is enough.
- Prefer deletion over addition, boring fixes, and the fewest files possible.
- For bugs, fix the shared root cause instead of patching one symptom path.
- If two small options exist, choose the edge-case-correct one.
- Mark intentional shortcuts with a `ponytail:` comment when they have a clear ceiling or upgrade path.

## Not Lazy About

Do not simplify away:

- input validation at trust boundaries
- error handling that prevents data loss
- security
- accessibility
- anything the user explicitly insists on
- the repo's Graphify-first rule

## Checks

Non-trivial logic needs one runnable check: the smallest test, compile command, or self-check that would fail if the logic breaks. Trivial one-liners do not need test ceremony.

## Intensity

- `lite`: build what was asked, but mention the lazier alternative.
- `full`: default; enforce the ladder and keep the diff short.
- `ultra`: challenge anything speculative, delete before adding.
