from unittest.mock import MagicMock

import pytest

from wallgen import backends
from wallgen import config as cfg
from wallgen.device import Capabilities, Placement

APPLE = Capabilities("Darwin", "arm64", False, 0.0, (0, 0), True, 16.0, True)
CUDA = Capabilities("Linux", "x86_64", True, 8.0, (8, 6), False, 32.0, False)


class _Stub:
    """Minimal backend stub that records the spec it was built for."""

    def __init__(self, name, fail_kinds=()):
        self.name = name
        self.fail_kinds = fail_kinds

    def __call__(self, config, spec, placement, lora_specs):
        if spec.kind in self.fail_kinds:
            raise backends.ModelUnsupported(f"{self.name} can't do {spec.kind}")
        b = MagicMock()
        b.name = self.name
        b.spec = spec
        b.device = placement.device if hasattr(placement, "device") else "mps"
        return b


def test_build_backend_selects_mlx_on_apple(monkeypatch, tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    monkeypatch.setattr(backends, "MLXBackend", _Stub("mlx"))
    monkeypatch.setattr(backends, "DiffusersBackend", _Stub("diffusers"))
    b = backends.build_backend(conf, "z-image-turbo", APPLE)
    assert b.name == "mlx" and b.spec.name == "z-image-turbo"


def test_build_backend_diffusers_off_apple(monkeypatch, tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    monkeypatch.setattr(backends, "MLXBackend", _Stub("mlx"))
    monkeypatch.setattr(backends, "DiffusersBackend", _Stub("diffusers"))
    b = backends.build_backend(conf, "z-image-turbo", CUDA)
    assert b.name == "diffusers"


def test_mlx_failure_falls_to_diffusers_same_model(monkeypatch, tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    monkeypatch.setattr(backends, "MLXBackend", _Stub("mlx", fail_kinds=("zimage",)))
    monkeypatch.setattr(backends, "DiffusersBackend", _Stub("diffusers"))
    b = backends.build_backend(conf, "z-image-turbo", APPLE)
    assert b.name == "diffusers" and b.spec.name == "z-image-turbo"  # same model, not fallback


def test_unsupported_model_falls_to_registry_fallback(monkeypatch, tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    monkeypatch.setattr(backends, "MLXBackend", _Stub("mlx"))
    # diffusers can't do zimage -> fall back to flux2-klein-4b (z-image's fallback).
    monkeypatch.setattr(backends, "DiffusersBackend", _Stub("diffusers", fail_kinds=("zimage",)))
    b = backends.build_backend(conf, "z-image-turbo", CUDA)
    assert b.spec.name == "flux2-klein-4b"


def test_mlx_generate_wiring():
    be = backends.MLXBackend.__new__(backends.MLXBackend)  # bypass heavy __init__
    be._model = MagicMock()
    be._model.generate_image.return_value = "IMG"
    out = be.generate("a prompt", 1024, 576, 9, 0.0, 42)
    assert out == "IMG"
    _, kw = be._model.generate_image.call_args
    assert kw["num_inference_steps"] == 9 and kw["width"] == 1024 and kw["seed"] == 42


def test_diffusers_place_offload_modes():
    pipe = MagicMock()
    backends.DiffusersBackend._place(pipe, Placement("cuda", "bfloat16", "Q5_K_M", "model"))
    pipe.enable_model_cpu_offload.assert_called_once()

    pipe2 = MagicMock()
    backends.DiffusersBackend._place(pipe2, Placement("cpu", "float32", "Q4_K_M", "sequential"))
    pipe2.enable_sequential_cpu_offload.assert_called_once()

    pipe3 = MagicMock()
    backends.DiffusersBackend._place(pipe3, Placement("mps", "bfloat16", "Q5_K_M", "none"))
    pipe3.to.assert_called_once_with("mps")
    pipe3.enable_model_cpu_offload.assert_not_called()
