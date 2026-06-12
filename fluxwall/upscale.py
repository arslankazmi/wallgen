"""High-resolution upscaling.

FLUX generates best at ~1024px; wallpapers want 1080p/1440p/4K. We generate at a
native size then upscale to the target. Real-ESRGAN (via realesrgan-ncnn-py, a
small Vulkan/Metal binary with no torch dependency) is the default; if it isn't
installed we fall back to a high-quality Lanczos resize so the pipeline still
produces a correctly-sized wallpaper.
"""

from __future__ import annotations

import logging

log = logging.getLogger("fluxwall.upscale")


def _lanczos(image, target: tuple[int, int]):
    from PIL import Image

    return image.resize(target, Image.LANCZOS)


def upscale_to(image, target: tuple[int, int], method: str = "realesrgan"):
    """Return ``image`` resized to ``target`` (w, h).

    Uses Real-ESRGAN when available and requested; otherwise Lanczos. Always
    finishes with an exact resize so the output matches the requested wallpaper
    resolution precisely.
    """
    tw, th = target
    if image.size == (tw, th):
        return image
    if method != "realesrgan":
        return _lanczos(image, target)

    try:
        from realesrgan_ncnn_py import Realesrgan
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("Real-ESRGAN unavailable (%s); using Lanczos resize", exc)
        return _lanczos(image, target)

    try:
        import numpy as np
        from PIL import Image

        # Choose an integer model scale that meets or exceeds the needed ratio.
        ratio = max(tw / image.width, th / image.height)
        model_scale = 4 if ratio > 2 else 2
        engine = Realesrgan(gpuid=0, model=0, scale=model_scale)
        upscaled = engine.process_pil(image.convert("RGB"))
        if not isinstance(upscaled, Image.Image):
            upscaled = Image.fromarray(np.asarray(upscaled))
        # Final exact fit to the target resolution.
        return upscaled.resize(target, Image.LANCZOS)
    except Exception as exc:  # noqa: BLE001
        log.warning("Real-ESRGAN failed (%s); using Lanczos resize", exc)
        return _lanczos(image, target)
