## MANDATORY: Graphify First Rule

**This is the highest-priority instruction in this file.**

Whenever the user asks a question about the codebase, architecture, modules, dependencies, or how something works — **YOU MUST run `graphify query "<the user's question>"` FIRST** before reading any source files, grepping, or using ripgrep.

**NO EXCEPTIONS except:**
1. The user explicitly says "do not use graphify"
2. The task is about fixing stale/incorrect graph output
3. `graphify-out/graph.json` does not exist

**Why:** Querying the graph costs ~40× fewer tokens than grepping raw files. If you skip graphify and start grepping, the user pays for every file you open. The graph gives you scoped, relevant files immediately.

**After graphify query:** Read only the files the graph points you to. Do not browse broadly.

**After code changes:** Run `graphify update .` to keep the graph current (AST-only, no API cost).

**Specific tools:**
- Broad questions: `graphify query "<question>"`
- Relationships between two things: `graphify path "<A>" "<B>"`
- Focused concept: `graphify explain "<concept>"`
- Only if graphify returns nothing useful: fall back to `rg` on the specific files it surfaced

## graphify (reference)

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use Graphify before doing anything else. If a dedicated `skill` tool is available, invoke it with `skill: "graphify"`; otherwise run the `graphify` CLI directly.

- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify.

## Vinted token-saving rules

- Use `docs/AI_CONTEXT.md` as the compact project map for future sessions.
- Do not scan `data/`, `runtime/`, or `graphify-out/` unless the user explicitly asks.
- Prefer `rg` or Graphify queries over opening broad documentation sets.
- Keep generated graph files local; this branch tracks the Graphify setup, not the generated graph payload.

## Ponytail coding rule — ALWAYS ACTIVE

Ponytail-style minimalism is **auto-activated on every session** via the `SessionStart` hook at `.claude/hooks/ponytail/ponytail-activate.js`. No trigger words needed.

Always use Ponytail for coding work: first ask whether the change needs to exist, then reuse existing repo code, stdlib, platform features, or installed dependencies before adding new code. Prefer deletion, boring fixes, and the fewest files possible. Do not simplify away trust-boundary validation, data-loss protection, security, accessibility, explicit user requirements, or the Graphify-first rule.

Source: `.agents/skills/ponytail/SKILL.md`.

## Caveman communication mode — ALWAYS ACTIVE

Caveman `full` mode is **auto-activated on every session** via the `SessionStart` hook at `.claude/hooks/caveman/caveman-activate.js`. The repo-local config `.caveman.json` pins the default to `full`, overriding any user-level config.

To switch levels: `/caveman lite|full|ultra`. To disable: `/caveman off` or "normal mode".

Source: `.agents/skills/caveman/SKILL.md`.
