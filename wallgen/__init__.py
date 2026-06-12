"""wallgen — self-hosted FLUX text-to-image wallpaper pipeline.

Privacy first: disable every external telemetry/analytics channel *before* any
third-party library (gradio, diffusers, transformers, huggingface_hub) is
imported anywhere in the process. Nothing here phones home.
"""

from __future__ import annotations

import os as _os

# Must run at import time, before gradio/HF libs are imported by any submodule.
_os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# Robust, high-performance model downloads (Xet) — large weights over a flaky
# connection otherwise stall mid-download. (hf_transfer is now deprecated.)
_os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
_os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # harmless on older hubs
# Keep the unified-memory watermark unbounded on Apple Silicon so large models
# can use available RAM instead of erroring at an artificial ceiling.
_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

__version__ = "0.1.0"

__all__ = ["__version__"]
