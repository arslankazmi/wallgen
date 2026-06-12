"""Prompt sources: inline strings, prompt files, and a deterministic daily
rotation for the scheduler. Pure-logic, no torch."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe slug from a prompt (used in output filenames)."""
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return slug[:max_len].strip("-") or "wallpaper"


def read_prompt_file(path: str | Path) -> list[str]:
    """Read a prompt-per-line file, skipping blanks and ``#`` comments."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    prompts: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def resolve_prompts(
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
) -> list[str]:
    """Build the batch of prompts from an inline prompt and/or a file."""
    out: list[str] = []
    if prompt:
        out.append(prompt)
    if prompt_file:
        out.extend(read_prompt_file(prompt_file))
    if not out:
        raise ValueError("No prompts provided (pass a prompt and/or a prompt file).")
    return out


def rotate_daily(prompts: list[str], day: _dt.date | None = None) -> str:
    """Deterministically pick one prompt for a given day so the scheduler cycles
    through the list without external state."""
    if not prompts:
        raise ValueError("Cannot rotate an empty prompt list.")
    day = day or _dt.date.today()
    index = day.toordinal() % len(prompts)
    return prompts[index]
