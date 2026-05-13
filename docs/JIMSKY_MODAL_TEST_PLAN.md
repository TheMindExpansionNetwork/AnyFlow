# AnyFlow Modal Test Plan

## Summary

AnyFlow is a good candidate for a **Modal video-diffusion budget/quality test harness**, not an immediate always-on production endpoint.

The upstream demo already exposes a useful experimental dial:

- T2V/TI2V/TV2V tasks through `demo.py`.
- Default demo output: 81 frames, 16 FPS, 480x832.
- Quick demo step count: 4 inference steps.
- Eval configs include multi-step sweeps: 2, 4, 8, 16, 32, and 50.

That maps well to Modal because we can run one bounded job, cache the model in a volume, collect MP4 artifacts, and shut down.

## Why this matters to Sonic-Forage / ORPHEUS

Use AnyFlow to answer these practical questions:

1. **Preview vs final quality:** Is 2-4 steps good enough for stream background previews?
2. **Cost envelope:** How much latency/cost do 8-16 steps add on Modal?
3. **Prompt stress test:** Do Sonic-Forage radio/world/skit prompts stay coherent in video form?
4. **Bridge lane:** Can we create quick motion cards before spending bigger GPU on Wan/LTX/HiDream pipelines?
5. **Multimodal skit support:** Can ORPHEUS skit timelines produce short motion beats for events like object drops, reactions, cleanup, and reveal moments?

## Recommended first Modal run order

### 1. CPU repo probe

No GPU, no checkpoint download.

```bash
modal run modal/anyflow_modal_probe.py --mode cpu-probe
```

Checks:

- Modal auth works.
- Repo files are present.
- Python can inspect configs.
- No paid GPU starts.

### 2. Dependency build probe

Build a slim inference image. Avoid full `requirements.txt` at first because it pulls heavy extras (`detectron2`, VBench, DreamSim) that are not needed for a minimal demo import.

Core packages to test first:

```text
torch / torchvision / torchaudio CUDA wheel
diffusers==0.38.0
transformers==4.50.0
accelerate==1.10.0
omegaconf==2.3.0
peft==0.17.0
imageio[ffmpeg]
opencv-python-headless
decord2
sentencepiece
huggingface_hub[cli]
```

### 3. 1.3B model cache probe

Download only:

```text
nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```

Store/cache in a Modal volume, not in git.

### 4. Tiny inference receipt

Run one prompt at 4 steps. Save:

```text
results/modal_smoke/<timestamp>/demo_t2v.mp4
results/modal_smoke/<timestamp>/receipt.json
```

Receipt should include:

- model id/path
- GPU class
- prompt
- width/height/frames/fps
- step count
- runtime seconds
- output path
- file size
- SHA-256
- whether any fallback happened

### 5. Any-step sweep

Run the same prompt at:

```text
2, 4, 8, 16, 32 steps
```

Then compare duration, cost, coherence, motion stability, and artifact level.

## First prompts to test

Use motion-rich but not impossible prompts:

```text
A neon pirate radio tower on an asteroid, antenna lights pulsing, camera slowly orbiting, cosmic dust drifting, cyberpunk broadcast energy, no text, no logos.
```

```text
A tiny cartoon character drops an ice cream cone on a glowing diner floor, the scoop splats, the character freezes in disbelief, cinematic comedic timing, no text, no logos.
```

```text
A futuristic AI radio studio comes alive at night, synth modules blinking, ghostly holographic DJs moving subtly, warm neon reflections, no text, no logos.
```

## Risks

- 14B checkpoints likely require expensive GPUs; do not start there.
- Full requirements include training/evaluation packages that may slow Modal image builds.
- Some upstream code assumes CUDA and fixed 480x832 / 81 frames; lower-size tests may require a small wrapper patch.
- VBench full evaluation is not a smoke test; keep it closed until model-load and single-inference pass.
- HF model access and download size should be checked before launching paid inference.

## Acceptance criteria for first successful Modal proof

- CPU Modal probe passes.
- Dependency image builds or the blocker is recorded with exact package error.
- 1.3B checkpoint is cached in Modal volume or download blocker is recorded.
- One MP4 is generated, downloaded or persisted, ffprobed, and sealed with SHA-256.
- Paid GPU runtime and cost class are documented.
- Modal app/job is not left running idle.
