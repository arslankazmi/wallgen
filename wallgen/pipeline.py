"""Generation orchestration: probe host -> pick model -> build backend ->
generate at native size -> stretch to the wallpaper target -> organise.
"""

from __future__ import annotations

import logging

from . import backends
from . import config as cfg
from . import device as dev
from . import loras as lora_mod
from . import organize
from .config import Config, LoraSpec

log = logging.getLogger("wallgen.pipeline")


class WallpaperGenerator:
    """Loads a backend once and generates one or more wallpapers from it."""

    def __init__(
        self,
        config: Config | None = None,
        profile_name: str | None = None,
        lora_specs: list[LoraSpec] | None = None,
    ):
        self.config = config or cfg.load_config()
        self.profile = self.config.profile(profile_name)
        self.lora_specs = lora_specs or []
        self.caps = dev.probe_capabilities()

        # Memory-aware model selection (VRAM + RAM on CUDA; RAM on MPS/CPU).
        self.model_name = self.config.fit_model(
            self.profile.model, ram_gb=self.caps.ram_gb, vram_gb=self.caps.vram_gb, has_cuda=self.caps.has_cuda
        )
        if self.profile.model not in ("auto", self.model_name):
            log.warning("%r doesn't fit this host; using %r", self.profile.model, self.model_name)

        self.backend = backends.build_backend(self.config, self.model_name, self.caps, self.lora_specs)
        self.spec = self.backend.spec
        # Steps/guidance: profile override else model-intrinsic default.
        self.steps = self.profile.steps if self.profile.steps is not None else self.spec.default_steps
        self.guidance = self.profile.guidance if self.profile.guidance is not None else self.spec.default_guidance
        log.info(
            "Profile %s | model=%s backend=%s quant=%s device=%s | steps=%d guidance=%s | loras=%d",
            self.profile.name, self.spec.name, self.backend.name, self.backend.quant,
            self.backend.device, self.steps, self.guidance, len(self.backend.active_loras),
        )

    def generate_one(self, prompt: str, seed: int | None = None) -> organize.GenerationRecord:
        gen_w, gen_h = self.config.gen_size
        target = self.config.target_size
        stretch = self.config.output.get("stretch", True)

        log.info("Generating %r @ %dx%d via %s", prompt, gen_w, gen_h, self.backend.name)
        image = self.backend.generate(prompt, gen_w, gen_h, self.steps, self.guidance, seed)

        stretched = False
        if stretch and tuple(image.size) != tuple(target):
            from PIL import Image

            image = image.resize(target, Image.LANCZOS)
            stretched = True
        final_size = tuple(image.size)

        png_path, sidecar_path = organize.target_paths(
            self.config.output_dir, prompt, self.config.output.get("organize_by", "date")
        )
        image.save(png_path)
        device = self.backend.device
        record = organize.GenerationRecord(
            prompt=prompt,
            model=self.spec.name,
            backend=self.backend.name,
            profile=self.profile.name,
            device=device,
            quant=self.backend.quant,
            steps=self.steps,
            guidance=self.guidance,
            seed=seed,
            gen_size=(gen_w, gen_h),
            target_size=final_size,
            stretched=stretched,
            loras=lora_mod.lora_metadata(self.backend.active_loras),
            created_at=organize.now_iso(),
            image_path=str(png_path),
        )
        organize.write_sidecar(sidecar_path, record)
        dev.empty_cache(device)
        log.info("Saved %s", png_path)
        return record

    def generate_many(self, prompts: list[str], seed: int | None = None) -> list[organize.GenerationRecord]:
        out = []
        for i, prompt in enumerate(prompts):
            out.append(self.generate_one(prompt, None if seed is None else seed + i))
        return out


def generate(
    prompts: list[str],
    profile_name: str | None = None,
    lora_stack: str | None = None,
    extra_loras: list[LoraSpec] | None = None,
    seed: int | None = None,
    config: Config | None = None,
) -> list[organize.GenerationRecord]:
    """Convenience one-shot: resolve loras, build a generator, run.

    Note: prompts here are taken as-is — prompt composition + lock handling are
    applied by the caller (CLI/scheduler) via ``prompts.build_prompts``.
    """
    config = config or cfg.load_config()
    specs = lora_mod.resolve_stack(config, lora_stack, extra_loras)
    runner = WallpaperGenerator(config, profile_name, specs)
    return runner.generate_many(prompts, seed)
