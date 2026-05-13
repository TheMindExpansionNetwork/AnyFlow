# AnyFlow Fork + Modal CPU Probe Receipt

## Summary

AnyFlow has been forked and prepared as an upstream-tracked Jimsky/MindExpander workbench for bounded Modal testing.

## Repos

- Upstream: https://github.com/NVlabs/AnyFlow
- Fork: https://github.com/TheMindExpansionNetwork/AnyFlow
- Local path: `/opt/data/workspace/github-forks/AnyFlow`
- Working branch: `jimsky/modal-skill`

## Upstream facts inspected

- Project: AnyFlow — Flow Map OPD for AnyStep Video Diffusion.
- License file: Apache License 2.0.
- Demo entrypoint: `demo.py`.
- Demo default inspected in code:
  - 480 x 832
  - 81 frames
  - 16 FPS output
  - 4 inference steps in quick demo path
- First recommended checkpoint for bounded testing:
  - `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`

## Added in this branch

- `AGENTS.md` — fork/agent safety notes.
- `docs/JIMSKY_FORK_STRATEGY.md` — branch model and upstream sync plan.
- `docs/JIMSKY_MODAL_TEST_PLAN.md` — CPU probe → dependency build → 1.3B cache → bounded inference sweep.
- `modal/anyflow_modal_probe.py` — safe CPU-only Modal probe.
- `scripts/jimsky-sync-upstream.sh` — conflict-stopping upstream sync helper.
- `docs/reports/anyflow_modal_cpu_probe.json` — successful Modal CPU probe output.

## Modal CPU probe

Command:

```bash
cd /opt/data/hermes-agent
source venv/bin/activate
set -a; source /opt/data/.env; set +a
cd /opt/data/workspace/github-forks/AnyFlow
modal run modal/anyflow_modal_probe.py --mode cpu-probe --output docs/reports/anyflow_modal_cpu_probe.json
```

Result:

- Modal app: `anyflow-modal-probe`
- Modal run URL: https://modal.com/apps/m1ndb0t-2045/main/ap-I6qBrcAT6PkXNOA4urjtNq
- CPU-only: yes
- GPU used: false
- Model downloaded: false
- Inference run: false
- Probe status: `ok: true`
- Container Python: `3.11.12`
- Container platform: `Linux-4.4.0-x86_64-with-glibc2.36`
- OmegaConf import: `2.3.0`

## How AnyFlow can help us test Modal

AnyFlow is useful for **dial-a-budget video diffusion testing**:

1. Start with a cheap Modal CPU/static probe.
2. Build a slim inference image without full VBench/detectron2 overhead.
3. Cache the 1.3B model in a Modal volume.
4. Generate one short T2V MP4 at 4 steps.
5. Sweep the same prompt at 2, 4, 8, 16, and 32 steps to measure quality vs latency/cost.

This directly supports Sonic-Forage / ORPHEUS use cases:

- stream background motion tests,
- skit/world moment video beats,
- prompt-to-video smoke tests,
- fast previews before spending bigger GPU on heavier video stacks.

## Closed gates

No paid GPU inference, 14B checkpoint, training, VBench evaluation, public media publishing, or persistent endpoint was started in this setup pass.

## Verification

- `python3 -m py_compile modal/anyflow_modal_probe.py` passed.
- Required doc/file needle checks passed.
- `git diff --check` passed before commit.
- Modal CPU probe passed and wrote JSON receipt.
