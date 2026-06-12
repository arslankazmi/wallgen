"""Configuration loading: profiles, model registry, resolution presets, LoRA stacks.

The on-disk format is ``config.yaml`` (see the repo root). This module parses it
into typed dataclasses and applies defaults so callers never deal with missing
keys. Pure-logic, no torch — fully unit-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("FLUXWALL_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "flux2" | "flux" | "sdxl"
    base: str
    gguf: str | None = None
    t5_gguf: str | None = None
    fallback: str | None = None  # name of another model to fall back to
    # Minimum total device memory (GiB) needed to run this model — GGUF + CPU
    # offload included. Used to auto-pick a model that fits the host.
    min_memory_gb: float = 0.0


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    quant: str = "auto"
    steps: int = 4
    guidance: float = 0.0
    max_sequence_length: int = 256


@dataclass(frozen=True)
class Resolution:
    name: str
    gen: tuple[int, int]  # native generation (w, h)
    target: tuple[int, int]  # final wallpaper (w, h) after upscale


@dataclass(frozen=True)
class LoraSpec:
    source: str  # HF repo id OR local path
    scale: float = 1.0


@dataclass(frozen=True)
class Config:
    default_profile: str
    profiles: dict[str, Profile]
    models: dict[str, ModelSpec]
    resolutions: dict[str, Resolution]
    lora_stacks: dict[str, list[LoraSpec]]
    generation: dict[str, Any]
    output: dict[str, Any]
    scheduler: dict[str, Any]
    source_path: Path | None = None

    def profile(self, name: str | None = None) -> Profile:
        key = name or self.default_profile
        if key not in self.profiles:
            raise KeyError(f"Unknown profile {key!r}. Available: {sorted(self.profiles)}")
        return self.profiles[key]

    def model(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(f"Unknown model {name!r}. Available: {sorted(self.models)}")
        return self.models[name]

    def resolution(self, name: str) -> Resolution:
        if name not in self.resolutions:
            raise KeyError(f"Unknown resolution {name!r}. Available: {sorted(self.resolutions)}")
        return self.resolutions[name]

    def lora_stack(self, name: str | None) -> list[LoraSpec]:
        if not name or name == "none":
            return []
        if name not in self.lora_stacks:
            raise KeyError(f"Unknown LoRA stack {name!r}. Available: {sorted(self.lora_stacks)}")
        return self.lora_stacks[name]

    def fit_model(self, requested: str, memory_gb: float) -> str:
        """Return a model that fits ``memory_gb``.

        If the requested model fits, it's used as-is. Otherwise the heaviest
        model that *does* fit is chosen (preferring the FLUX family over the
        SDXL fallback) so a low-memory host (e.g. a 16 GB Mac asking for the
        12B prod tier) transparently runs the smaller FLUX model instead of
        OOMing. If nothing is marked as fitting, the request is returned
        unchanged (the loader will offload and try its best).
        """
        spec = self.model(requested)
        if memory_gb >= spec.min_memory_gb:
            return requested
        candidates = [m for m in self.models.values() if m.min_memory_gb <= memory_gb]
        if not candidates:
            return requested
        # Prefer FLUX kinds, then the largest model that still fits.
        best = max(candidates, key=lambda m: (m.kind in ("flux2", "flux"), m.min_memory_gb))
        return best.name

    @property
    def output_dir(self) -> Path:
        return Path(self.output.get("dir", "./output"))


_DEFAULTS: dict[str, Any] = {
    "default_profile": "dev",
    "profiles": {
        "dev": {"model": "flux2-klein-4b", "quant": "auto", "steps": 4, "guidance": 0.0},
        "prod": {"model": "flux1-schnell", "quant": "auto", "steps": 4, "guidance": 0.0},
    },
    "models": {
        "flux2-klein-4b": {
            "kind": "flux2",
            "base": "black-forest-labs/FLUX.2-klein-4B",
            "gguf": "unsloth/FLUX.2-klein-4B-GGUF",
            "fallback": "sdxl-lightning",
            "min_memory_gb": 12,  # 4B GGUF + offloaded encoder fits a 16 GB Mac
        },
        "flux1-schnell": {
            "kind": "flux",
            "base": "black-forest-labs/FLUX.1-schnell",
            "gguf": "city96/FLUX.1-schnell-gguf",
            "t5_gguf": "city96/t5-v1_1-xxl-encoder-gguf",
            "min_memory_gb": 24,  # 12B + T5-XXL; needs a roomier host
        },
        "sdxl-lightning": {"kind": "sdxl", "base": "stabilityai/sdxl-turbo", "min_memory_gb": 8},
    },
    "resolutions": {
        "hd": {"gen": [1024, 576], "target": [1920, 1080]},
        "qhd": {"gen": [1024, 576], "target": [2560, 1440]},
        "uhd": {"gen": [1024, 576], "target": [3840, 2160]},
        "square": {"gen": [1024, 1024], "target": [2048, 2048]},
    },
    "generation": {"upscale": True, "upscaler": "realesrgan", "batch_size": 1, "seed": None},
    "output": {"dir": "./output", "organize_by": "date"},
    "scheduler": {
        "profile": "prod",
        "resolution": "uhd",
        "theme_rotation": "prompts/wallpapers.txt",
        "set_desktop": True,
        "loras": [],
    },
    "loras": {"none": []},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def parse_loras(raw: Any) -> list[LoraSpec]:
    """Parse a YAML LoRA list (items may be ``"source"`` strings or
    ``{source, scale}`` dicts) into :class:`LoraSpec`s. Shared by the config
    loader and the scheduler so the format has a single source of truth."""
    specs: list[LoraSpec] = []
    for item in raw or []:
        if isinstance(item, str):
            specs.append(LoraSpec(source=item))
        elif isinstance(item, dict):
            specs.append(LoraSpec(source=item["source"], scale=float(item.get("scale", 1.0))))
    return specs


# Backwards-compatible private alias.
_parse_loras = parse_loras


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML (merged over built-in defaults). Missing file is OK
    — defaults are used."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    merged = _deep_merge(_DEFAULTS, raw)

    profiles = {
        name: Profile(
            name=name,
            model=p["model"],
            quant=p.get("quant", "auto"),
            steps=int(p.get("steps", 4)),
            guidance=float(p.get("guidance", 0.0)),
            max_sequence_length=int(p.get("max_sequence_length", 256)),
        )
        for name, p in merged["profiles"].items()
    }
    models = {
        name: ModelSpec(
            name=name,
            kind=m["kind"],
            base=m["base"],
            gguf=m.get("gguf"),
            t5_gguf=m.get("t5_gguf"),
            fallback=m.get("fallback"),
            min_memory_gb=float(m.get("min_memory_gb", 0.0)),
        )
        for name, m in merged["models"].items()
    }
    resolutions = {
        name: Resolution(
            name=name,
            gen=(int(r["gen"][0]), int(r["gen"][1])),
            target=(int(r["target"][0]), int(r["target"][1])),
        )
        for name, r in merged["resolutions"].items()
    }
    lora_stacks = {name: _parse_loras(items) for name, items in merged.get("loras", {}).items()}

    return Config(
        default_profile=merged["default_profile"],
        profiles=profiles,
        models=models,
        resolutions=resolutions,
        lora_stacks=lora_stacks,
        generation=merged["generation"],
        output=merged["output"],
        scheduler=merged["scheduler"],
        source_path=cfg_path if cfg_path.exists() else None,
    )
