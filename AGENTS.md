# Agent Notes for TheMindExpansionNetwork/AnyFlow

## Fork Context

- Origin fork: https://github.com/TheMindExpansionNetwork/AnyFlow
- Upstream source: https://github.com/NVlabs/AnyFlow
- Local path: `/opt/data/workspace/github-forks/AnyFlow`
- Upstream push URL must stay disabled (`upstream DISABLED`) so agents cannot accidentally push to NVlabs.
- License: Apache-2.0 in `LICENSE`; preserve attribution, copyright notices, and upstream citation text.

## Branch Model

- `main`: keep aligned with upstream `NVlabs/AnyFlow/main` when possible.
- `jimsky/modal-skill`: Jimsky/MindExpander Modal test, docs, and Hermes skill integration lane.
- `vendor-sync/*`: upstream merge/rebase branches only; never destructive reset over Jimsky work.

## Safety Rules

- Do not commit `.env`, tokens, Hugging Face credentials, Modal tokens, SSH keys, private datasets, private prompts, or generated media batches.
- Do not commit model weights/checkpoints or downloaded Hugging Face model directories; keep them in Modal volumes, HF cache volumes, or local ignored directories.
- Do not launch paid GPU runs unless the operator explicitly asks for a bounded run. CPU Modal probes and syntax checks are allowed.
- Prefer small Modal probes first: repo import/static inspection, dependency install smoke, model-card/download metadata check, then a short 1.3B inference only after approval.
- Keep custom wrappers under `modal/`, `scripts/`, and `docs/` where possible so upstream code remains easy to sync.

## AnyFlow Fit for Sonic-Forage / ORPHEUS

AnyFlow is useful as a video-generation evaluation workbench because it exposes any-step video diffusion behavior over Wan-style models. The practical questions for our stack are:

1. Can Modal build and cache the dependency image reliably?
2. Can a smaller AnyFlow/Wan 1.3B checkpoint load on an affordable GPU class?
3. How do quality/latency change across 2, 4, 8, 16, and 32 inference steps?
4. Can we use it as a fast test lane for stream visuals, skit/world loops, and prompt-to-video smoke tests before heavier A100/H100 jobs?

Record all Modal runs in `docs/reports/` with command, app/function, GPU class, duration, output artifact path, and whether any paid GPU was used.
