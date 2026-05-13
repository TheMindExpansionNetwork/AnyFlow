# Copyright 2026 The Mind Expansion Network
# SPDX-License-Identifier: Apache-2.0
"""Bounded AnyFlow Modal TV2V smoke on old footage.

This is intentionally a one-shot queued test harness, not a persistent endpoint.
It takes a short local MP4, uploads the bytes as a Modal function argument,
loads the AnyFlow FAR Wan 1.3B checkpoint, runs a small text+video-to-video
smoke, returns the MP4 bytes, and writes a JSON receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "anyflow-tv2v-smoke"
MODEL_ID = "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = Path("/workspace/AnyFlow")
REMOTE_MODEL_DIR = Path("/models/AnyFlow-FAR-Wan2.1-1.3B-Diffusers")
REMOTE_OUTPUT_DIR = Path("/outputs/anyflow-tv2v-smoke")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("anyflow-models", create_if_missing=True)
output_volume = modal.Volume.from_name("anyflow-outputs", create_if_missing=True)

# Build a slim inference image rather than installing full requirements.txt
# (detectron2/VBench/DreamSim are evaluation/training extras and slow the smoke lane).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate==1.10.0",
        "decord2",
        "diffusers==0.38.0",
        "einops==0.8.1",
        "ftfy",
        "huggingface_hub[cli]",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "numpy<2.0.0",
        "omegaconf==2.3.0",
        "opencv-python-headless",
        "peft==0.17.0",
        "sentencepiece",
        "transformers==4.50.0",
        "regex",
        "tqdm",
    )
    .add_local_dir(str(REPO_ROOT), str(REMOTE_REPO))
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=49152,
    timeout=60 * 60,
    volumes={"/models": model_volume, "/outputs": output_volume},
)
def run_tv2v_smoke(
    input_video: bytes,
    input_name: str,
    prompt: str,
    steps: int = 4,
    frames: int = 81,
    width: int = 832,
    height: int = 480,
    seed: int = 333,
) -> dict[str, Any]:
    started = time.time()
    os.chdir(REMOTE_REPO)
    repo_path = str(REMOTE_REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    os.environ["PYTHONPATH"] = f"{repo_path}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("HF_HOME", "/models/hf-home")
    os.environ.setdefault("HF_HUB_CACHE", "/models/hf-cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = Path("/tmp") / f"anyflow_tv2v_{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / input_name
    input_path.write_bytes(input_video)

    import decord  # type: ignore
    import torch
    from diffusers.utils import export_to_video
    from huggingface_hub import snapshot_download
    from PIL import Image  # noqa: F401 - imported to match upstream demo deps
    from torchvision import transforms

    from far.pipelines.pipeline_far_wan_anyflow import FARWanAnyFlowPipeline
    from far.utils.video_util import select_frame_indices
    from far.utils.vis_util import draw_rectangle

    decord.bridge.set_bridge("torch")

    if not REMOTE_MODEL_DIR.exists() or not any(REMOTE_MODEL_DIR.iterdir()):
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(REMOTE_MODEL_DIR),
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        )
        model_volume.commit()

    torch.cuda.empty_cache()
    pipe = FARWanAnyFlowPipeline.from_pretrained(str(REMOTE_MODEL_DIR)).to("cuda", dtype=torch.bfloat16)

    num_cond_frames = 25
    video_reader = decord.VideoReader(str(input_path))
    frame_idxs = select_frame_indices(len(video_reader), video_reader.get_avg_fps(), target_fps=16)[:num_cond_frames]
    cond_frames = video_reader.get_batch(frame_idxs)
    cond_frames = (cond_frames / 255.0).float().permute(0, 3, 1, 2).contiguous()
    cond_frames = transforms.Resize([height, width])(cond_frames).unsqueeze(0)
    context_sequence, context_length = {"raw": cond_frames}, cond_frames.shape[1]

    generator = torch.Generator("cuda").manual_seed(seed)
    video = pipe(
        prompt=prompt,
        context_sequence=context_sequence,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        generator=generator,
    ).frames[0]
    video = draw_rectangle(video, context_length=context_length)

    out_dir = REMOTE_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"anyflow_tv2v_{steps}steps_{Path(input_name).stem}.mp4"
    export_to_video(video, output_video_path=str(out_path), fps=16)

    output_bytes = out_path.read_bytes()
    receipt = {
        "ok": True,
        "app": APP_NAME,
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 3),
        "gpu": "L40S",
        "model_id": MODEL_ID,
        "input_name": input_name,
        "input_sha256": _sha256_bytes(input_video),
        "input_ffprobe": _ffprobe(input_path),
        "prompt": prompt,
        "steps": steps,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": 16,
        "seed": seed,
        "output_remote_path": str(out_path),
        "output_size_bytes": len(output_bytes),
        "output_sha256": _sha256_path(out_path),
        "output_ffprobe": _ffprobe(out_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    output_volume.commit()
    return {"receipt": receipt, "video_bytes": output_bytes}


@app.local_entrypoint()
def main(
    input_path: str,
    output_dir: str = "/opt/data/drops/anyflow-modal-smoke/results",
    prompt: str = "Transform this old footage into a neon cosmic Sonic-Forage pirate radio world, glowing broadcast energy, surreal sci-fi motion, cinematic camera continuity, no text, no logos.",
    steps: int = 4,
    frames: int = 81,
    seed: int = 333,
) -> None:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(src)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if frames != 81:
        print(
            f"Requested frames={frames}, but AnyFlow FAR 1.3B checkpoint chunk_partition sums to 21 latent frames, "
            "which requires 81 video frames. Overriding frames=81 for this smoke lane.",
            flush=True,
        )
        frames = 81
    print(f"Uploading input: {src} ({src.stat().st_size} bytes)", flush=True)
    print(f"Prompt: {prompt}", flush=True)
    print(f"Running AnyFlow TV2V smoke: steps={steps} frames={frames} seed={seed}", flush=True)
    result = run_tv2v_smoke.remote(
        src.read_bytes(),
        src.name,
        prompt,
        steps=steps,
        frames=frames,
        seed=seed,
    )
    receipt = result["receipt"]
    local_mp4 = out_root / Path(receipt["output_remote_path"]).name
    local_json = out_root / (local_mp4.stem + ".receipt.json")
    local_mp4.write_bytes(result["video_bytes"])
    local_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    print(f"LOCAL_MP4={local_mp4}", flush=True)
    print(f"LOCAL_RECEIPT={local_json}", flush=True)
