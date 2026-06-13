# Claude Code Tips & Tricks

Notes from "32 Tricks to Level Up Claude Code in 16 Mins" (https://www.youtube.com/watch?v=jqoFP9QapXI)

## Beginner Hacks

1. **Run `/init` on every project.** Claude Code scans the codebase and generates a `CLAUDE.md` cheat sheet covering architecture, conventions, and key files, so you don't have to re-explain the project every session. For a new project, describe the goal, tech stack, and key folders/rules and have Claude build the `CLAUDE.md` for you.

2. **Set up a status line.** Run `/statusline` and tell Claude what you want shown (model, context %, cost, etc.). It generates a small script that acts like a mini dashboard at the bottom of your terminal — useful for watching context usage and avoiding context rot.

3. **Use voice input.** Claude Code has a native `/voice` command (rolling out) so you can talk directly to your terminal and have it code for you. Alternatively, use a voice-to-text app to dictate prompts anywhere.

4. **Keep your context small.** Don't dump the entire codebase into a conversation — only give Claude what it needs for the current task. Break big problems into small, focused steps. Less noise in the context window = better performance.

5. **Use `/context` to find token bloat.** Shows a breakdown (by percentage) of what's eating your tokens — system prompts, file contents, MCP servers, etc. — so you can diagnose and restructure a bloated session.

6. **Compact at ~60% and clear between tasks.** Run `/compact` around 60% context usage to compress conversation history without losing important info. You can guide it: *"compact, but keep all the API integration decisions and database schema."* When switching to a totally different task, use `/clear` to start fresh — your `CLAUDE.md` and files still preserve project context.

7. **Always start in plan mode.** Toggle with Shift+Tab. In plan mode, Claude can read and research but can't make changes — it outlines steps, asks clarifying questions, and maps an approach before writing code. Once you approve the plan, switch out and tell it to execute. This dramatically reduces correction cycles.

8. **Treat Claude like a junior developer.** Instead of direct commands ("write me a function that does X"), give it problems to reason through ("how should we handle growth tracking?"). Letting it make and explain its own assumptions/decisions leads to better outputs.

9. **Make Claude ask questions.** Explicitly tell it to invoke the "ask user question" tool: *"Continuously ask me questions until you're 95% confident you understand exactly what I need and what you need to do."* This front-loads alignment and avoids multiple rounds of revisions.

10. **Build self-checking into to-do lists.** Add verification steps directly into Claude's generated to-do list — e.g., after "build the website," add "take a screenshot and check it looks right" and "open Chrome DevTools and confirm no errors." Tell it not to move to the next to-do until it's ~95% confident the current one is good.

## Intermediate Hacks

11. **Deploy sub-agents for parallel work.** Ask the main session to use sub-agents for complex problems. Each sub-agent gets its own context window (and can use its own model), works in parallel (research, tests, exploring approaches), and reports back to the main thread — like having a team of developers.

12. **Build custom skills.** Create reusable prompt files in `.claude/skills/` (e.g., `techdebt.md`, `codereview.md`) that encode a full workflow. Invoke via natural language or a `/command`. Commit them to the repo so your whole team can use them — effectively automating your SOPs.

13. **Use Haiku for sub-agents.** For simple tasks or processing large amounts of data, set sub-agents to use Haiku instead of Opus. E.g., have a Haiku sub-agent scrape/read hundreds of thousands of tokens of articles and hand back a small summary to the (more expensive) main agent. Keeps costs down without sacrificing quality where it matters.

14. **Constantly refresh your `CLAUDE.md`.** Whenever there's a new discovery, pattern, gotcha, or convention, update `CLAUDE.md` so Claude starts the next session already knowing it. But keep it lean (~150–200 lines max) since it's loaded into every conversation as part of the system prompt — trim it if it grows too large.

15. **Have `CLAUDE.md` route to other files.** Instead of cramming everything in, link out to separate files for style guides, business context, reference docs, etc. `CLAUDE.md` just needs to tell Claude *where* to look, not contain everything itself — saves tokens on info that isn't always needed.

16. **Exit early and re-ask.** If Claude starts going down the wrong path, don't wait for it to finish — hit Escape, correct course, and re-prompt immediately. Every token spent going the wrong direction is wasted context. Steer tight, steer early.

17. **Challenge outputs aggressively.** If an output is just "okay," push back: *"Scrap that, do a more elegant version"* or *"try again with a completely different approach."* Claude often produces a dramatically better second attempt with a higher bar set. Once it improves, tell it to update the relevant skill/`CLAUDE.md` so it doesn't repeat the mistake.

18. **Use `/rewind` for quick undos.** Rolls the conversation back to a previous point without starting over — fast and clean recovery from a wrong turn.

19. **Use hooks for notifications.** Run `/hooks` (or describe it in natural language) to set up a notification — e.g., a sound when a session/chat finishes. Useful for running many parallel sessions and knowing when one needs your input.

20. **Use screenshots.** Claude can see images — feed it error messages, inspiration websites, or screenshots of your own app. Build a self-check loop: *"take a screenshot and tell me if the layout looks right,"* then iterate (design → screenshot → revise) multiple passes before the first version even reaches you.

21. **Use Chrome DevTools.** Claude can open a browser, interact with an app, fill out forms, and check functionality — like the screenshot loop but for actual app behavior, not just visuals. Huge for front-end work and for tasks without an explicit API.

22. **Clone inspiration sites.** Screenshot a site you like and feed it to Claude with "make it look like this" — it recreates the design patterns without generic "AI slop." You can also feed in actual HTML/styling as a template and add your own touches.

## Advanced Hacks

23. **Run parallel sessions with Git worktrees.** Use `claude --worktree <feature-name>` to create an isolated workspace on its own branch — run multiple sessions on the same project simultaneously without them overwriting each other's work. Merge branches back together when done.

24. **Use API endpoints instead of MCP servers (when appropriate).** MCP servers load all their tool definitions into the context window, which costs tokens. If you only need one specific operation (e.g., reading one Notion database), hardcode that API endpoint directly instead of loading the whole MCP server.

25. **Use `/loop` for recurring tasks.** E.g., *"every 5 minutes check on the deployment"* — Claude reruns the prompt on that interval within the session, only interrupting when something needs attention. Also works for one-time natural-language reminders. Note: loops only last ~3 days; for longer-term automation, use desktop scheduled tasks (though those run as individual sessions without shared context memory).

26. **Host on a VPS for always-on sessions.** Run Claude Code on a remote server so it stays running when your laptop is closed — SSH in anytime, or interact via Telegram. Great for long-running tasks you don't want to babysit locally.

27. **Use remote control from your phone.** Claude Code now supports controlling local sessions from your phone or browser — start a task locally, walk away, and keep steering it remotely. Code stays on your local machine; only the control connection is remote.

28. **No-SQL data analytics.** Connect CLI tools (e.g., BigQuery's `bq`) to Claude Code and ask questions in plain English (*"what were our top 10 revenue sources last quarter?"*) — Claude translates to the right query, runs it, and gives you the answer. Works for any CLI-based tool.

29. **Ultra think.** For hard problems (architecture decisions, complex debugging, big refactors, or when normal prompts aren't giving the right output), type "ultra think" — it allocates a much larger thinking budget (~32,000 tokens) before responding. Don't use it for simple fixes; use it for system-wide decisions or after a couple of failed attempts.

30. **Edit permissions for safe autonomy instead of `--dangerously-skip-permissions`.** Explicitly allow commands you know are safe, and explicitly deny destructive ones (deletes, removes, etc.) — deny always takes priority over allow. Gets you the speed/autonomy of skipping permissions without the risk.

31. **Use agent teams.** Unlike isolated sub-agents (fresh context, no cross-talk), agent teams can communicate with each other, share a task list, and assign each other work — and you can talk to individual agents directly, not just the main one. More expensive and slower, but more cohesive output for big projects.

32. **Context7 MCP.** Install the Context7 MCP server to pull current, version-specific documentation and live code examples for popular libraries (Next.js, React, MongoDB, etc.) directly into the conversation before Claude writes code. Solves the "training data cutoff" problem of Claude suggesting renamed/deprecated/nonexistent APIs.
