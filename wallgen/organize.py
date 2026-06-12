"""Output organisation: where wallpapers land and the metadata sidecar written
beside each one. Pure-logic (filesystem + json only)."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .prompts import slugify


@dataclass
class GenerationRecord:
    """Everything needed to reproduce a wallpaper. Serialised to the JSON sidecar."""

    prompt: str  # the composed prompt actually sent to the model
    model: str
    backend: str  # "mlx" | "diffusers"
    profile: str
    device: str
    quant: str
    steps: int
    guidance: float
    seed: int | None
    gen_size: tuple[int, int]
    target_size: tuple[int, int]
    stretched: bool
    loras: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def output_subdir(base_dir: str | Path, organize_by: str = "date", day: _dt.date | None = None) -> Path:
    base = Path(base_dir)
    if organize_by == "date":
        day = day or _dt.date.today()
        return base / day.isoformat()
    return base


def _unique_path(directory: Path, slug: str, suffix: str) -> Path:
    candidate = directory / f"{slug}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{slug}-{counter}{suffix}"
        counter += 1
    return candidate


def target_paths(base_dir: str | Path, prompt: str, organize_by: str = "date") -> tuple[Path, Path]:
    """Return (png_path, json_sidecar_path), creating the directory."""
    directory = output_subdir(base_dir, organize_by)
    directory.mkdir(parents=True, exist_ok=True)
    png = _unique_path(directory, slugify(prompt), ".png")
    return png, png.with_suffix(".json")


def write_sidecar(sidecar_path: str | Path, record: GenerationRecord) -> None:
    Path(sidecar_path).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")
