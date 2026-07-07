#!/usr/bin/env node
// ponytail — Claude Code SessionStart activation hook
//
// Runs on every session start to auto-activate Ponytail coding minimalism.
// No trigger words or manual invocation needed — always on.
//
// Reads SKILL.md from the repo source of truth so edits propagate automatically.

const fs = require('fs');
const path = require('path');

// Resolve SKILL.md locations — try both .claude/skills/ and .agents/skills/
const candidates = [
  path.join(__dirname, '..', 'skills', 'ponytail', 'SKILL.md'),
  path.join(process.cwd(), '.agents', 'skills', 'ponytail', 'SKILL.md'),
  path.join(__dirname, '..', '..', '.agents', 'skills', 'ponytail', 'SKILL.md')
];

let skillContent = '';
for (const p of candidates) {
  try {
    skillContent = fs.readFileSync(p, 'utf8');
    break;
  } catch (e) { /* try next */ }
}

let output;

if (skillContent) {
  // Strip YAML frontmatter
  const body = skillContent.replace(/^---[\s\S]*?---\s*/, '');
  output = 'PONYTAIL MODE ACTIVE (full). Auto-on every session.\n\n' + body;
} else {
  // Fallback ruleset when SKILL.md is not found
  output =
    'PONYTAIL MODE ACTIVE (full). Auto-on every session.\n\n' +
    'Lazy senior developer mode. Lazy = efficient, not careless.\n\n' +
    '## Ladder (stop at first rung that works)\n\n' +
    '1. Does this need to exist at all?\n' +
    '2. Does this repo already have a helper, pattern, or module for it?\n' +
    '3. Does the standard library do it?\n' +
    '4. Does the platform or framework already do it?\n' +
    '5. Does an already-installed dependency do it?\n' +
    '6. Can it be one line?\n' +
    '7. Only then, write the minimum code that works.\n\n' +
    '## Rules\n\n' +
    '- No speculative abstractions, factories, wrappers, config, or scaffolding.\n' +
    '- No new dependency when a small local change or existing dependency is enough.\n' +
    '- Prefer deletion over addition, boring fixes, and the fewest files possible.\n' +
    '- For bugs, fix the shared root cause instead of patching one symptom path.\n\n' +
    '## Not Lazy About (do not simplify away)\n\n' +
    '- input validation at trust boundaries\n' +
    '- error handling that prevents data loss\n' +
    '- security\n' +
    '- accessibility\n' +
    '- anything the user explicitly insists on\n' +
    '- the repo\'s Graphify-first rule';
}

process.stdout.write(output);