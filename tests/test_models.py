"""Tests for the loader's fallback + placement logic that don't require torch."""

from unittest.mock import MagicMock

import pytest

from fluxwall import config as cfg
from fluxwall import models
from fluxwall.device import DeviceProfile

CPU = DeviceProfile(device="cpu", dtype="float32", total_memory_gb=8.0, quant="Q4_K_M")


def _config_with_cycle(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
default_profile: dev
models:
  a: { kind: bogus, base: x, fallback: b }
  b: { kind: bogus, base: y, fallback: a }
""".strip()
    )
    return cfg.load_config(p)


def test_fallback_cycle_terminates(tmp_path):
    """A -> B -> A fallback loop must raise, not recurse forever."""
    conf = _config_with_cycle(tmp_path)
    with pytest.raises(models.ModelUnsupported):
        models.load_pipeline(conf, "a", CPU)


def test_unknown_kind_without_fallback_raises(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("models:\n  solo: { kind: bogus, base: x }\n")
    conf = cfg.load_config(p)
    with pytest.raises(models.ModelUnsupported):
        models.load_pipeline(conf, "solo", CPU)


def test_place_mps_never_offloads():
    # Unified memory: offload duplicates weights, so MPS always loads resident.
    pipe = MagicMock()
    models._place(pipe, DeviceProfile("mps", "bfloat16", 16.0, "Q4_K_M"), heavy=True)
    pipe.enable_model_cpu_offload.assert_not_called()
    pipe.to.assert_called_once_with("mps")


def test_place_cuda_offloads_only_heavy_low_vram():
    # Heavy + low VRAM CUDA -> offload (VRAM is separate from RAM).
    pipe = MagicMock()
    models._place(pipe, DeviceProfile("cuda", "bfloat16", 12.0, "Q5_K_M"), heavy=True)
    pipe.enable_model_cpu_offload.assert_called_once()

    # Heavy + high VRAM CUDA -> resident.
    pipe2 = MagicMock()
    models._place(pipe2, DeviceProfile("cuda", "bfloat16", 80.0, "Q8_0"), heavy=True)
    pipe2.enable_model_cpu_offload.assert_not_called()
    pipe2.to.assert_called_once_with("cuda")

    # Non-heavy (sdxl) -> resident regardless.
    pipe3 = MagicMock()
    models._place(pipe3, DeviceProfile("cuda", "bfloat16", 12.0, "Q5_K_M"), heavy=False)
    pipe3.enable_model_cpu_offload.assert_not_called()
    pipe3.to.assert_called_once_with("cuda")
