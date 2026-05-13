# Copyright 2026 The Mind Expansion Network
# SPDX-License-Identifier: Apache-2.0
"""Modal probes for the AnyFlow fork.

Default mode is intentionally CPU-only: it verifies Modal auth, container startup,
repo availability, and AnyFlow config discovery without starting paid GPU work.

Usage from the repository root after loading Modal credentials:

    modal run modal/anyflow_modal_probe.py --mode cpu-probe

The GPU/model-load lanes are documented but left closed by default. Add a small
inference function only after the CPU probe and dependency image build are known
good and the operator approves a bounded GPU run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "anyflow-modal-probe"
REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("omegaconf==2.3.0", "pyyaml")
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_repo(root: Path) -> dict[str, Any]:
    interesting = [
        "README.md",
        "requirements.txt",
        "demo.py",
        "far/main.py",
        "options/test/anyflow/test_AnyFlow-Wan2.1-T2V-1.3B-Diffusers.yml",
        "options/test/anyflow/test_AnyFlow-FAR-Wan2.1-1.3B-Diffusers.yml",
    ]
    files: dict[str, Any] = {}
    for rel in interesting:
        p = root / rel
        files[rel] = {
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else None,
            "sha256": _sha256(p) if p.exists() and p.is_file() else None,
        }

    test_cfg_dir = root / "options" / "test" / "anyflow"
    train_cfg_dir = root / "options" / "train" / "anyflow"
    return {
        "root": str(root),
        "files": files,
        "test_config_count": len(list(test_cfg_dir.glob("*.yml"))) if test_cfg_dir.exists() else 0,
        "train_config_count": len(list(train_cfg_dir.rglob("*.yml"))) if train_cfg_dir.exists() else 0,
    }


@app.function(
    image=cpu_image,
    timeout=180,
    cpu=1.0,
    memory=1024,
)
def cpu_repo_probe(repo_payload: dict[str, str]) -> dict[str, Any]:
    """CPU-only Modal probe. No GPU, no model download, no inference."""
    root = Path("/tmp/anyflow_probe_repo")
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in repo_payload.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    try:
        import omegaconf  # type: ignore

        omegaconf_version = getattr(omegaconf, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        omegaconf_version = f"import_failed: {exc}"

    report = {
        "ok": True,
        "app": APP_NAME,
        "mode": "cpu-probe",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omegaconf_version": omegaconf_version,
        "repo": _inspect_repo(root),
        "gpu_used": False,
        "model_downloaded": False,
        "inference_run": False,
    }
    return report


def _payload_from_local_repo() -> dict[str, str]:
    rels = [
        "README.md",
        "requirements.txt",
        "demo.py",
        "far/main.py",
        "options/test/anyflow/test_AnyFlow-Wan2.1-T2V-1.3B-Diffusers.yml",
        "options/test/anyflow/test_AnyFlow-FAR-Wan2.1-1.3B-Diffusers.yml",
    ]
    payload: dict[str, str] = {}
    for rel in rels:
        p = REPO_ROOT / rel
        if p.exists():
            payload[rel] = p.read_text(errors="replace")
    return payload


@app.local_entrypoint()
def main(mode: str = "cpu-probe", output: str = "") -> None:
    if mode != "cpu-probe":
        raise ValueError(
            "Only mode=cpu-probe is implemented safely. "
            "Use docs/JIMSKY_MODAL_TEST_PLAN.md before adding GPU/model lanes."
        )
    report = cpu_repo_probe.remote(_payload_from_local_repo())
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}", flush=True)
