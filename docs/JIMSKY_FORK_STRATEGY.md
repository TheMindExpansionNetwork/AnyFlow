# Jimsky Fork Strategy for AnyFlow

## Purpose

This fork keeps NVIDIA's AnyFlow repo available as an upstream-compatible workbench while we explore whether it can become a bounded Modal video-diffusion test lane for Sonic-Forage, ORPHEUS-333, and livestream/world-generation experiments.

## Repositories

- Upstream: https://github.com/NVlabs/AnyFlow
- Fork: https://github.com/TheMindExpansionNetwork/AnyFlow
- Local clone: `/opt/data/workspace/github-forks/AnyFlow`

## What AnyFlow appears to provide

Based on the upstream README and code inspection:

- Any-step video diffusion framework based on flow maps.
- Wan2.1-based AnyFlow checkpoints at 1.3B and 14B scales.
- Demo modes:
  - `t2v`: text-to-video
  - `ti2v`: text+image-to-video
  - `tv2v`: text+video-to-video
- Demo defaults in `demo.py`:
  - 480 x 832 output
  - 81 frames
  - 16 FPS export
  - 4 inference steps in the quick demo path
- Evaluation configs sweep step counts such as 2, 4, 8, 16, 32, and 50.

The main value for us is **quality/latency testing across arbitrary inference-step budgets**. That maps directly to cost-conscious Modal experiments.

## Branch model

- `main`: upstream-aligned fork branch.
- `jimsky/modal-skill`: Modal wrappers, docs, and Hermes skill integration.
- `jimsky/*`: future product/integration branches.
- `vendor-sync/*`: conflict-safe upstream sync branches.

Do not use `git reset --hard upstream/main` on a branch with Jimsky commits.

## Safe sync recipe

```bash
cd /opt/data/workspace/github-forks/AnyFlow
git fetch upstream --prune
git fetch origin --prune
git checkout main
git merge --ff-only upstream/main
git push origin main
git checkout jimsky/modal-skill
git rebase main
git push --force-with-lease origin jimsky/modal-skill
```

If the rebase conflicts, stop and report; do not auto-resolve model/config conflicts.

## Modal strategy

### Phase 0: CPU/static probe

Goal: prove Modal credentials, repo checkout/copy, Python syntax/static inspection, and environment metadata without installing the full GPU stack.

Command:

```bash
cd /opt/data/hermes-agent
source venv/bin/activate
set -a; source /opt/data/.env; set +a
modal run /opt/data/workspace/github-forks/AnyFlow/modal/anyflow_modal_probe.py --mode cpu-probe
```

Expected: JSON with CUDA visibility, repo files, detected configs, and no GPU spend.

### Phase 1: dependency image build probe

Goal: build a Modal image with PyTorch CUDA, diffusers, transformers, decord, and core dependencies. Avoid VBench/detectron2 until needed because they can make builds slow and brittle.

### Phase 2: model download/cache probe

Goal: download only the 1.3B checkpoint into a Modal volume/HF cache and record model size and files. Do not run inference yet.

Recommended model for first test:

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```

### Phase 3: bounded inference smoke

Goal: one short T2V output with 1.3B checkpoint, 4 steps, 81 frames at 480x832 if memory allows. If not, patch a lower-frame/resolution test wrapper rather than pretending upstream demo supports it.

Suggested Modal GPU order:

1. A10G or L40S for image/dependency/model-load probes.
2. A100/H100 only if 1.3B inference OOMs or 14B is explicitly approved.

### Phase 4: cost/quality sweep

Run the same prompt at 2, 4, 8, 16, and 32 steps, store MP4s and a receipt. This directly answers whether AnyFlow gives us a useful “dial-a-budget” video lane.

## What this can help us test

- Whether Modal can cheaply host a fast video-diffusion smoke lane.
- Whether AnyFlow's any-step property gives useful preview/final tradeoffs for livestream visuals.
- Whether ORPHEUS skit/video prompts can get quick moving clips before heavier Wan/LTX/HiDream jobs.
- Whether generated videos are stable enough for radio-loop backgrounds and world-drop motion cards.

## Closed gates

Do not run:

- 14B inference,
- training,
- VBench full evaluation,
- multi-GPU torchrun,
- public publishing of generated media,
- or persistent paid endpoints

without explicit operator approval and a bounded run receipt.
