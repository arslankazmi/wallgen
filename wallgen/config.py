"""Configuration: model registry (per-backend sources), profiles, output sizing,
prompt directive + locking, and LoRA stacks.

Parses ``config.yaml`` into typed dataclasses over built-in defaults. Pure-logic,
no torch — fully unit-testable.
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
    kind: str  # "zimage" | "flux2" | "sdxl"
    priority: int = 0  # higher = preferred when several models fit the host
    min_memory_gb: float = 0.0  # total quantized footprint (GiB) — binds on MPS/CPU + CUDA RAM staging
    min_vram_gb: float = 0.0  # on-device transformer footprint (GiB) — binds CUDA VRAM (encoder offloads to RAM)
    default_steps: int = 4
    default_guidance: float = 0.0
    fallback: str | None = None  # model to fall back to if this one can't load
    # diffusers backend sources
    base: str | None = None
    gguf: str | None = None  # GGUF transformer repo
    text_encoder_gguf: str | None = None  # GGUF text-encoder repo (Qwen3 for zimage/flux2)
    # mlx (mflux) backend source
    mlx: dict[str, Any] | None = None  # {repo?, base_model, quant}


@dataclass(frozen=True)
class Profile:
    name: str
    model: str  # a model name, or "auto" for memory-aware selection
    quant: str = "auto"
    steps: int | None = None  # None -> use the model's default_steps
    guidance: float | None = None  # None -> use the model's default_guidance


@dataclass(frozen=True)
class LoraSpec:
    source: str  # HF repo id OR local path
    scale: float = 1.0


@dataclass(frozen=True)
class Config:
    default_profile: str
    profiles: dict[str, Profile]
    models: dict[str, ModelSpec]
    output: dict[str, Any]
    prompt: dict[str, Any]
    scheduler: dict[str, Any]
    lora_stacks: dict[str, list[LoraSpec]]
    source_path: Path | None = None

    # ---- lookups -------------------------------------------------------------
    def profile(self, name: str | None = None) -> Profile:
        key = name or self.default_profile
        if key not in self.profiles:
            raise KeyError(f"Unknown profile {key!r}. Available: {sorted(self.profiles)}")
        return self.profiles[key]

    def model(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(f"Unknown model {name!r}. Available: {sorted(self.models)}")
        return self.models[name]

    def lora_stack(self, name: str | None) -> list[LoraSpec]:
        if not name or name == "none":
            return []
        if name not in self.lora_stacks:
            raise KeyError(f"Unknown LoRA stack {name!r}. Available: {sorted(self.lora_stacks)}")
        return self.lora_stacks[name]

    # ---- sizing --------------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return Path(self.output.get("dir", "./output"))

    @property
    def gen_size(self) -> tuple[int, int]:
        w, h = self.output.get("gen_size", [1024, 576])
        return int(w), int(h)

    @property
    def target_size(self) -> tuple[int, int]:
        w, h = self.output.get("target_size", [1920, 1080])
        return int(w), int(h)

    # ---- memory-aware model selection ---------------------------------------
    def fit_model(self, requested: str, ram_gb: float, vram_gb: float = 0.0, has_cuda: bool = False) -> str:
        """Resolve a model name against the host's memory.

        On CUDA a model fits iff its transformer fits VRAM **and** its total
        footprint fits system RAM (the encoder is offloaded to RAM). On MPS/CPU
        only the unified/system RAM total matters. ``requested == "auto"`` (or a
        model that doesn't fit) -> the highest-``priority`` model that fits; an
        explicit, fitting request is honoured as-is.
        """

        def fits(spec: ModelSpec) -> bool:
            if has_cuda:
                return vram_gb >= spec.min_vram_gb and ram_gb >= spec.min_memory_gb
            return ram_gb >= spec.min_memory_gb

        if requested != "auto":
            spec = self.model(requested)
            if fits(spec):
                return requested
        candidates = [m for m in self.models.values() if fits(m)]
        if not candidates:
            # Nothing fits cleanly: smallest model + let the backend offload hard.
            return min(self.models.values(), key=lambda m: m.min_memory_gb).name
        return max(candidates, key=lambda m: m.priority).name

    # ---- prompt composition + locking ---------------------------------------
    def compose_prompt(self, user_prompt: str) -> str:
        """Apply the wallpaper directive/template to a user prompt."""
        directive = self.prompt.get("wallpaper_directive", "")
        tw, th = self.target_size
        directive = directive.format(target_w=tw, target_h=th)
        template = self.prompt.get("lock", {}).get("template", "{prompt}, {wallpaper_directive}")
        if not directive:
            return user_prompt
        return template.format(prompt=user_prompt, wallpaper_directive=directive)

    @property
    def lock_mode(self) -> str:
        # YAML coerces bare ``off``/``on`` to booleans — normalise back to strings.
        m = self.prompt.get("lock", {}).get("mode", "off")
        if m is False:
            return "off"
        if m is True:
            return "on"
        return str(m)

    @property
    def locked_prompts_path(self) -> str:
        return self.prompt.get("lock", {}).get("locked_prompts", "prompts/wallpapers.txt")


# --- defaults ----------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "default_profile": "dev",
    "profiles": {"dev": {"model": "auto"}, "prod": {"model": "auto"}},
    "models": {
        "z-image-turbo": {
            "kind": "zimage",
            "priority": 30,
            "min_memory_gb": 10,
            "min_vram_gb": 7,
            "default_steps": 9,
            "default_guidance": 0.0,
            "mlx": {"repo": "filipstrand/Z-Image-Turbo-mflux-4bit", "base_model": "z-image-turbo", "quant": 4},
            "base": "Tongyi-MAI/Z-Image-Turbo",
            "gguf": "unsloth/Z-Image-Turbo-GGUF",
            "text_encoder_gguf": "unsloth/Qwen3-4B-GGUF",
            "fallback": "flux2-klein-4b",
        },
        "flux2-klein-4b": {
            "kind": "flux2",
            "priority": 20,
            "min_memory_gb": 8,
            "min_vram_gb": 4,
            "default_steps": 4,
            "default_guidance": 0.0,
            "mlx": {"repo": "Runpod/FLUX.2-klein-4B-mflux-4bit", "base_model": "flux2-klein-4b", "quant": 4},
            "base": "black-forest-labs/FLUX.2-klein-4B",
            "gguf": "unsloth/FLUX.2-klein-4B-GGUF",
            "text_encoder_gguf": "unsloth/Qwen3-4B-GGUF",
            "fallback": "sd-turbo",
        },
        "sd-turbo": {
            "kind": "sdxl",
            "priority": 10,
            "min_memory_gb": 4,
            "min_vram_gb": 4,
            "default_steps": 2,
            "default_guidance": 0.0,
            "base": "stabilityai/sd-turbo",
        },
    },
    "output": {
        "gen_size": [1024, 576],
        "target_size": [1920, 1080],
        "stretch": True,
        "dir": "./output",
        "organize_by": "date",
    },
    "prompt": {
        "wallpaper_directive": (
            "high-quality desktop wallpaper, full-bleed edge-to-edge composition for a "
            "{target_w}x{target_h} screen; the image will be stretched to fill the screen, "
            "so use balanced framing with no critical detail near the edges; "
            "no text, no watermark, no border"
        ),
        "lock": {
            "mode": "off",
            "template": "{prompt}, {wallpaper_directive}",
            "locked_prompts": "prompts/wallpapers.txt",
        },
    },
    "scheduler": {"profile": "prod", "lock_mode": "locked", "set_desktop": True, "loras": []},
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
    """Parse a YAML LoRA list (``"source"`` strings or ``{source, scale}`` dicts).
    Shared by the config loader and the scheduler — one source of truth."""
    specs: list[LoraSpec] = []
    for item in raw or []:
        if isinstance(item, str):
            specs.append(LoraSpec(source=item))
        elif isinstance(item, dict):
            specs.append(LoraSpec(source=item["source"], scale=float(item.get("scale", 1.0))))
    return specs


def _parse_model(name: str, m: dict[str, Any]) -> ModelSpec:
    # Sources may be nested under `diffusers:`/`mlx:` or given flat.
    diff = m.get("diffusers", {})
    return ModelSpec(
        name=name,
        kind=m["kind"],
        priority=int(m.get("priority", 0)),
        min_memory_gb=float(m.get("min_memory_gb", 0.0)),
        min_vram_gb=float(m.get("min_vram_gb", 0.0)),
        default_steps=int(m.get("default_steps", 4)),
        default_guidance=float(m.get("default_guidance", 0.0)),
        fallback=m.get("fallback"),
        base=diff.get("base", m.get("base")),
        gguf=diff.get("gguf", m.get("gguf")),
        text_encoder_gguf=diff.get("text_encoder_gguf", m.get("text_encoder_gguf")),
        mlx=m.get("mlx"),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML (merged over built-in defaults). Missing file -> defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    merged = _deep_merge(_DEFAULTS, raw)

    profiles = {
        name: Profile(
            name=name,
            model=p.get("model", "auto"),
            quant=p.get("quant", "auto"),
            steps=p.get("steps"),
            guidance=p.get("guidance"),
        )
        for name, p in merged["profiles"].items()
    }
    models = {name: _parse_model(name, m) for name, m in merged["models"].items()}
    lora_stacks = {name: parse_loras(items) for name, items in merged.get("loras", {}).items()}

    return Config(
        default_profile=merged["default_profile"],
        profiles=profiles,
        models=models,
        output=merged["output"],
        prompt=merged["prompt"],
        scheduler=merged["scheduler"],
        lora_stacks=lora_stacks,
        source_path=cfg_path if cfg_path.exists() else None,
    )
