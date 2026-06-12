"""Model registry + unified pipeline loader.

One entry point — :func:`load_pipeline` — turns a :class:`~fluxwall.config.ModelSpec`
plus a resolved :class:`~fluxwall.device.DeviceProfile` into a ready-to-run
Diffusers pipeline, regardless of whether the model is FLUX.2, FLUX.1 (GGUF), or
an SDXL fallback.

Design notes:
* GGUF quantization is the only quant path that works on both MPS and CUDA, so it
  is the default for the FLUX tiers. FP8/NF4 are CUDA-only and deliberately unused.
* If the installed ``diffusers`` lacks FLUX.2 support, the loader raises
  :class:`ModelUnsupported` and the caller transparently falls back to the model's
  configured ``fallback`` (SDXL-Lightning by default).
* When LoRA adapters are requested, GGUF is skipped in favour of a bf16 base,
  because adapter loading/fusing is unreliable on quantized weights (see loras.py).
"""

from __future__ import annotations

import logging
from typing import Any

from .config import Config, ModelSpec
from .device import DeviceProfile, torch_dtype

log = logging.getLogger("fluxwall.models")


class ModelUnsupported(RuntimeError):
    """Raised when the installed diffusers cannot load the requested model kind."""


def _resolve_gguf_file(repo: str, quant: str) -> str:
    """Download and return the local path of the GGUF file in ``repo`` matching
    ``quant`` (e.g. "Q5_K_M"). Picks the smallest match if several exist."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    matches = [f for f in files if quant.lower() in f.lower()]
    if not matches:
        raise ModelUnsupported(
            f"No GGUF file matching quant {quant!r} in {repo!r}. Available: {files}"
        )
    matches.sort(key=len)
    chosen = matches[0]
    log.info("Using GGUF %s from %s", chosen, repo)
    return hf_hub_download(repo, chosen)


# Below this much device memory (GiB), the heavy FLUX text encoders won't fit
# resident, so we stream components on/off the device via CPU offload.
_OFFLOAD_MEMORY_GB = 24.0


def _place(pipe, profile: DeviceProfile, heavy: bool = False):
    """Move/offload the pipeline onto the active device and enable memory savers.

    Placement strategy differs by backend:

    * **CUDA** — VRAM is separate from system RAM, so CPU offload genuinely frees
      VRAM. The heavy FLUX tiers offload on cards below ~24 GB.
    * **MPS / Apple Silicon** — memory is *unified*: "CPU" and "device" are the
      same physical pool, so ``enable_model_cpu_offload`` only duplicates weights
      and wastes RAM. We always load resident and lean on attention slicing +
      VAE tiling to keep activation memory down (model selection via
      ``Config.fit_model`` already ensures the weights themselves fit).
    """
    needs_offload = (
        heavy and profile.device == "cuda" and profile.total_memory_gb < _OFFLOAD_MEMORY_GB
    )
    if needs_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001 - fall back to resident placement
            pipe.to(profile.device)
    else:
        pipe.to(profile.device)
    # Keep activation memory low (important for the resident MPS path + high-res).
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def _load_flux2(spec: ModelSpec, profile: DeviceProfile, use_gguf: bool):
    # Probe FLUX.2 support (klein has a dedicated pipeline class; we load via the
    # base repo's model_index so the right one is auto-resolved).
    try:
        from diffusers import DiffusionPipeline, Flux2Transformer2DModel  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure means unsupported
        raise ModelUnsupported(f"diffusers has no FLUX.2 support: {exc}") from exc

    dtype = torch_dtype(profile.dtype)
    if use_gguf and spec.gguf:
        try:
            from diffusers import GGUFQuantizationConfig  # type: ignore

            ckpt = _resolve_gguf_file(spec.gguf, profile.quant)
            transformer = Flux2Transformer2DModel.from_single_file(
                ckpt,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            pipe = DiffusionPipeline.from_pretrained(spec.base, transformer=transformer, torch_dtype=dtype)
        except ModelUnsupported:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("FLUX.2 GGUF load failed (%s); loading bf16 base instead", exc)
            pipe = DiffusionPipeline.from_pretrained(spec.base, torch_dtype=dtype)
    else:
        pipe = DiffusionPipeline.from_pretrained(spec.base, torch_dtype=dtype)
    return _place(pipe, profile, heavy=True)


def _load_flux(spec: ModelSpec, profile: DeviceProfile, use_gguf: bool):
    from diffusers import FluxPipeline, FluxTransformer2DModel  # type: ignore

    dtype = torch_dtype(profile.dtype)
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_gguf and spec.gguf:
        try:
            from diffusers import GGUFQuantizationConfig  # type: ignore

            ckpt = _resolve_gguf_file(spec.gguf, profile.quant)
            kwargs["transformer"] = FluxTransformer2DModel.from_single_file(
                ckpt,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001 - GGUF missing/incompatible
            log.warning("FLUX GGUF load failed (%s); loading bf16 base instead", exc)
            kwargs.pop("transformer", None)
    # NOTE: spec.t5_gguf is reserved for a future quantized-T5 path to shave
    # encoder memory on the prod tier; not yet wired (base T5 is used for now).
    pipe = FluxPipeline.from_pretrained(spec.base, **kwargs)
    return _place(pipe, profile, heavy=True)


def _load_sdxl(spec: ModelSpec, profile: DeviceProfile):
    from diffusers import AutoPipelineForText2Image  # type: ignore

    dtype = torch_dtype(profile.dtype)
    pipe = AutoPipelineForText2Image.from_pretrained(spec.base, torch_dtype=dtype)
    return _place(pipe, profile)


def load_pipeline(
    config: Config,
    model_name: str,
    profile: DeviceProfile,
    use_loras: bool = False,
    _visited: frozenset[str] | None = None,
):
    """Load (and place) the pipeline for ``model_name``.

    Returns ``(pipeline, effective_model_spec, used_gguf)``. Falls back to the
    model's configured ``fallback`` if its kind is unsupported by the installed
    diffusers. GGUF is skipped when ``use_loras`` is True. ``_visited`` guards
    against fallback cycles (A -> B -> A) in misconfigured registries.
    """
    visited = (_visited or frozenset()) | {model_name}
    spec = config.model(model_name)
    # LoRA adapters attach reliably only to unquantized weights.
    use_gguf = not use_loras

    try:
        if spec.kind == "flux2":
            return _load_flux2(spec, profile, use_gguf), spec, use_gguf
        if spec.kind == "flux":
            return _load_flux(spec, profile, use_gguf), spec, use_gguf
        if spec.kind == "sdxl":
            return _load_sdxl(spec, profile), spec, False
        raise ModelUnsupported(f"Unknown model kind {spec.kind!r}")
    except ModelUnsupported as exc:
        if spec.fallback and spec.fallback in config.models and spec.fallback not in visited:
            log.warning("%s unsupported (%s); falling back to %s", model_name, exc, spec.fallback)
            return load_pipeline(config, spec.fallback, profile, use_loras=use_loras, _visited=visited)
        raise
