from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


LATEST_RUN_FILE = "latest_run.txt"


def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    return cleaned.strip("-") or "run"


def create_run_name(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _sanitize_run_name(f"{prefix}_{stamp}")


def _runs_root(base_dir: Path) -> Path:
    return base_dir / "runs"


def _latest_file(base_dir: Path) -> Path:
    return base_dir / LATEST_RUN_FILE


def set_latest_run(base_dir: Path, run_name: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    _latest_file(base_dir).write_text(run_name, encoding="utf-8")


def get_latest_run(base_dir: Path) -> Optional[str]:
    latest = _latest_file(base_dir)
    if latest.exists():
        value = latest.read_text(encoding="utf-8").strip()
        if value:
            return value

    runs_dir = _runs_root(base_dir)
    if not runs_dir.exists():
        return None

    candidates = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    return sorted(candidates)[-1] if candidates else None


def resolve_run_dir(base_dir: str | Path, run_name: Optional[str] = None) -> Path:
    base = Path(base_dir)
    if (base / "config_snapshot.yaml").exists():
        return base

    chosen = run_name or get_latest_run(base)
    candidate = _runs_root(base) / chosen if chosen else None
    if candidate and candidate.exists():
        return candidate

    if (base / "metrics.json").exists():
        return base

    return candidate if candidate is not None else base


def prepare_run_dirs(cfg: dict, resume: bool = False, run_name: Optional[str] = None):
    log_root = Path(cfg["logging"]["log_dir"])
    model_root = Path(cfg["logging"]["model_dir"])

    if resume:
        chosen = run_name or get_latest_run(log_root) or get_latest_run(model_root)
        if not chosen:
            raise FileNotFoundError("No previous run found to resume.")
    else:
        chosen = _sanitize_run_name(run_name) if run_name else create_run_name("train")

    log_dir = _runs_root(log_root) / chosen
    model_dir = _runs_root(model_root) / chosen
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_latest_run(log_root, chosen)
    set_latest_run(model_root, chosen)
    return chosen, log_root, model_root, log_dir, model_dir


def save_run_snapshot(cfg: dict, run_name: str, log_dir: str | Path, model_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    snapshot = deepcopy(cfg)
    snapshot["logging"]["run_name"] = run_name
    snapshot["logging"]["resolved_log_dir"] = str(log_dir)
    snapshot["logging"]["resolved_model_dir"] = str(model_dir)

    with open(log_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)

    meta = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "model_dir": str(model_dir),
    }
    with open(log_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
