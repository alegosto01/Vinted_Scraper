# Ponytail Workflow

Ponytail is this repo's minimal-implementation rule for coding work. It is based on `DietrichGebert/ponytail`, but this repo keeps a lightweight local copy instead of vendoring the external plugin runtime.

## Quick Start

Use Ponytail when a task asks for the simplest correct implementation:

```text
Use Ponytail. Add the smallest safe change for ...
```

or:

```text
/ponytail
```

## Ladder

Before writing code, stop at the first rung that works:

1. Does this need to exist?
2. Does this repo already have a helper, util, or pattern?
3. Does the standard library do it?
4. Does the platform or framework do it?
5. Does an installed dependency do it?
6. Can it be one line?
7. Only then, write the minimum code that works.

## Repo Wiring

- `AGENTS.md` has the always-on coding rule.
- `.agents/skills/ponytail/SKILL.md` is the Codex/repo skill.
- `.kimi/skills/ponytail/SKILL.md` mirrors the Graphify-style Kimi skill.
- `skills-lock.json` records the upstream source and local skill hash.

## Boundaries

Ponytail does not override:

- Graphify-first for codebase questions
- input validation at trust boundaries
- error handling that prevents data loss
- security
- accessibility
- explicit user requirements

Non-trivial logic still needs the smallest runnable check.
