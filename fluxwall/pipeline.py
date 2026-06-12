"""Generation orchestration: profile -> device/quant -> load -> (loras) -> loop
-> upscale -> organise. This is the single place the heavy pieces come together.
"""

from __future__ import annotations

import logging

from . import config as cfg
from . import device as dev
from . import loras as lora_mod
from . import models, organize, upscale
from .config import Config, LoraSpec

log = logging.getLogger("fluxwall.pipeline")

# Model kinds that accept FLUX's `max_sequence_length` argument.
_FLUX_KINDS = {"flux", "flux2"}


class WallpaperGenerator:
    """Loads a pipeline once and generates one or more wallpapers from it."""

    def __init__(
        self,
        config: Config | None = None,
        profile_name: str | None = None,
        lora_specs: list[LoraSpec] | None = None,
    ):
        self.config = config or cfg.load_config()
        self.profile = self.config.profile(profile_name)
        self.lora_specs = lora_specs or []
        self.device = dev.resolve_profile(self.profile.quant)
        # Memory-aware model selection: downgrade to a model that fits the host
        # (e.g. 16 GB Mac asking for the 12B prod tier -> the 4B FLUX.2-klein).
        self.model_name = self.config.fit_model(self.profile.model, self.device.total_memory_gb)
        if self.model_name != self.profile.model:
            log.warning(
                "Host has %.1fGB; %r needs more, using %r instead.",
                self.device.total_memory_gb, self.profile.model, self.model_name,
            )
        log.info(
            "Profile %s | model=%s | device=%s dtype=%s quant=%s mem=%.1fGB | loras=%d",
            self.profile.name, self.model_name, self.device.device, self.device.dtype,
            self.device.quant, self.device.total_memory_gb, len(self.lora_specs),
        )
        self.pipe, self.spec, self.used_gguf = models.load_pipeline(
            self.config, self.model_name, self.device, use_loras=bool(self.lora_specs)
        )
        # `active_loras` are the adapters that actually loaded — recorded in the
        # sidecar so it reflects reality, not merely what was requested.
        self.active_loras = lora_mod.apply_loras(self.pipe, self.lora_specs)

    def _make_generator(self, seed: int | None):
        if seed is None:
            return None
        import torch

        # CPU generator is safest across backends for reproducibility.
        return torch.Generator("cpu").manual_seed(int(seed))

    def generate_one(
        self,
        prompt: str,
        resolution_name: str = "hd",
        seed: int | None = None,
        do_upscale: bool | None = None,
    ) -> organize.GenerationRecord:
        res = self.config.resolution(resolution_name)
        gen_w, gen_h = res.gen
        do_upscale = self.config.generation.get("upscale", True) if do_upscale is None else do_upscale

        call_kwargs = dict(
            prompt=prompt,
            width=gen_w,
            height=gen_h,
            num_inference_steps=self.profile.steps,
            guidance_scale=self.profile.guidance,
            generator=self._make_generator(seed),
        )
        if self.spec.kind in _FLUX_KINDS:
            call_kwargs["max_sequence_length"] = self.profile.max_sequence_length

        log.info("Generating %r @ %dx%d (%s)", prompt, gen_w, gen_h, self.spec.name)
        image = self.pipe(**call_kwargs).images[0]

        if do_upscale:
            image = upscale.upscale_to(
                image, res.target, method=self.config.generation.get("upscaler", "realesrgan")
            )
            final_size = res.target
            upscaled = True
        else:
            final_size = image.size
            upscaled = False

        png_path, sidecar_path = organize.target_paths(
            self.config.output_dir, prompt, self.config.output.get("organize_by", "date")
        )
        image.save(png_path)
        record = organize.GenerationRecord(
            prompt=prompt,
            model=self.spec.name,
            profile=self.profile.name,
            device=self.device.device,
            dtype=self.device.dtype,
            quant=self.device.quant if self.used_gguf else "none",
            steps=self.profile.steps,
            guidance=self.profile.guidance,
            seed=seed,
            gen_size=(gen_w, gen_h),
            target_size=tuple(final_size),
            upscaled=upscaled,
            loras=lora_mod.lora_metadata(self.active_loras),
            created_at=organize.now_iso(),
            image_path=str(png_path),
        )
        organize.write_sidecar(sidecar_path, record)
        dev.empty_cache(self.device.device)
        log.info("Saved %s", png_path)
        return record

    def generate_many(
        self,
        prompts: list[str],
        resolution_name: str = "hd",
        seed: int | None = None,
        do_upscale: bool | None = None,
    ) -> list[organize.GenerationRecord]:
        records = []
        for i, prompt in enumerate(prompts):
            s = None if seed is None else seed + i
            records.append(self.generate_one(prompt, resolution_name, s, do_upscale))
        return records


def generate(
    prompts: list[str],
    profile_name: str | None = None,
    resolution_name: str = "hd",
    lora_stack: str | None = None,
    extra_loras: list[LoraSpec] | None = None,
    seed: int | None = None,
    do_upscale: bool | None = None,
    config: Config | None = None,
) -> list[organize.GenerationRecord]:
    """Convenience one-shot: resolve config + loras, build a generator, run."""
    config = config or cfg.load_config()
    specs = lora_mod.resolve_stack(config, lora_stack, extra_loras)
    runner = WallpaperGenerator(config, profile_name, specs)
    return runner.generate_many(prompts, resolution_name, seed, do_upscale)
