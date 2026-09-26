# CLAUDE.md

## Purpose
This is a **learning project**: I'm building RAG from scratch to understand it deeply, not just to get working code. Understanding comes before speed.

## How to work with me
- **Go step by step.** Add code gradually, following the natural flow of a RAG pipeline (load → chunk → embed → store → retrieve → generate → improve). Never jump ahead or build several stages at once.
- **Small increments.** Each change should be small enough to read and understand in a few minutes. If a step is big, split it and check in with me.
- **Explain the why.** Before writing code, say briefly what we're adding, why it's needed, and where it fits in the pipeline. After writing it, point out the key lines.
- **Show how it connects.** When a new script or module appears, explain how it relates to the existing ones (what it takes in, what it produces, who calls it).
- **Make it runnable.** Every step should end with something I can run and see output from, so I can observe what changed.
- **Prefer plain code over magic.** Implement core ideas by hand first. Introduce a library only after I understand what it replaces, and say what it's doing under the hood.
- **Don't do my thinking for me.** When there's a design choice (chunk size, similarity metric, prompt format), lay out the options and tradeoffs and let me choose.
- **Ask, don't assume.** If you're unsure what I want next, ask instead of guessing.
- don't be too pedant - it should move relatively quick.

## Code style
- Simple, readable, well-named. No premature abstraction.
- Comments explain *why*, not *what*.
- Keep dependencies minimal.

## Keep track
Maintain `PROGRESS.md` with a short list of what we've built so far, what each file does, and what's next. Update it after each step.
