---
name: ponytail
description: "Lazy senior developer mode: use the smallest correct implementation. Trigger when the user says ponytail, lazy mode, minimal solution, YAGNI, do less, or asks to avoid over-engineering."
trigger: /ponytail
---

# /ponytail

Use Ponytail mode for coding work.

Before writing code, stop at the first rung that works:

1. Does this need to exist?
2. Does this repo already have a helper or pattern?
3. Does the standard library do it?
4. Does the platform/framework do it?
5. Does an installed dependency do it?
6. Can it be one line?
7. Only then, write the minimum code that works.

Do not skip understanding: read the touched flow first. Do not simplify away validation, data-loss protection, security, accessibility, explicit user requirements, or the repo's Graphify-first rule.

For non-trivial logic, leave one runnable check.
