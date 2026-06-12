"""Captioned-image dataset loading for LoRA finetuning.

Two layouts are supported (both produced by the ak:ml-curate-dataset workflow):
  1. paired files:  ``<name>.png`` + ``<name>.txt`` (caption) in one directory
  2. metadata file: ``metadata.jsonl`` with ``{"file_name": ..., "text": ...}``

This module only *reads and validates* the dataset; the actual training is run
by ``train_lora.run`` via the external finetune workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class Example:
    image_path: Path
    caption: str


def _from_metadata(meta: Path) -> list[Example]:
    examples: list[Example] = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(Example(meta.parent / row["file_name"], row.get("text", "")))
    return examples


def _from_pairs(directory: Path) -> list[Example]:
    examples: list[Example] = []
    for img in sorted(directory.iterdir()):
        if img.suffix.lower() in _IMAGE_EXTS:
            caption_file = img.with_suffix(".txt")
            caption = caption_file.read_text(encoding="utf-8").strip() if caption_file.exists() else ""
            examples.append(Example(img, caption))
    return examples


def load_dataset(directory: str | Path) -> list[Example]:
    """Load + validate a captioned-image dataset from ``directory``."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {d}")
    meta = d / "metadata.jsonl"
    examples = _from_metadata(meta) if meta.exists() else _from_pairs(d)
    if not examples:
        raise ValueError(f"No captioned images found in {d}")
    missing = [str(e.image_path) for e in examples if not e.image_path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset references missing images: {missing[:5]}")
    return examples
