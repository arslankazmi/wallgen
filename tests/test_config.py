import pytest

from wallgen import config as cfg


def test_defaults(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    assert conf.default_profile == "dev"
    assert conf.profile().model == "auto"
    assert set(conf.models) == {"z-image-turbo", "flux2-klein-4b", "sd-turbo"}
    z = conf.model("z-image-turbo")
    assert z.kind == "zimage" and z.priority == 30 and z.mlx and z.default_steps == 9
    assert z.base == "Tongyi-MAI/Z-Image-Turbo"
    assert conf.gen_size == (1024, 576)
    assert conf.target_size == (1920, 1080)


def test_fit_model_per_host(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    # M4 16GB unified (mps) -> z-image (best that fits RAM).
    assert conf.fit_model("auto", ram_gb=16) == "z-image-turbo"
    # RTX 3060 Ti: 8GB VRAM, 32GB RAM -> z-image (vram 7<=8, total 10<=32).
    assert conf.fit_model("auto", ram_gb=32, vram_gb=8, has_cuda=True) == "z-image-turbo"
    # GTX 1660 Ti: 6GB VRAM, 16GB RAM -> klein (z-image needs 7GB VRAM).
    assert conf.fit_model("auto", ram_gb=16, vram_gb=6, has_cuda=True) == "flux2-klein-4b"
    # i3 no-GPU 8GB -> klein (z-image needs 10GB RAM).
    assert conf.fit_model("auto", ram_gb=8) == "flux2-klein-4b"
    # i3 no-GPU 4GB -> sd-turbo floor.
    assert conf.fit_model("auto", ram_gb=4) == "sd-turbo"
    # Explicit fitting request honoured.
    assert conf.fit_model("sd-turbo", ram_gb=64) == "sd-turbo"


def test_nested_source_parsing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
models:
  custom:
    kind: zimage
    priority: 5
    min_memory_gb: 6
    diffusers: { base: org/model, gguf: org/model-gguf, text_encoder_gguf: org/enc-gguf }
    mlx: { repo: org/model-mlx, base_model: custom, quant: 8 }
""".strip()
    )
    conf = cfg.load_config(p)
    m = conf.model("custom")
    assert m.base == "org/model" and m.gguf == "org/model-gguf"
    assert m.text_encoder_gguf == "org/enc-gguf"
    assert m.mlx["repo"] == "org/model-mlx"
    # Defaults still merged in.
    assert "z-image-turbo" in conf.models


def test_compose_prompt_and_lock(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
prompt:
  wallpaper_directive: "wallpaper for {target_w}x{target_h}, stretched"
  lock: { mode: template, template: "{prompt} :: {wallpaper_directive}" }
output: { target_size: [2560, 1440] }
""".strip()
    )
    conf = cfg.load_config(p)
    assert conf.lock_mode == "template"
    composed = conf.compose_prompt("a lake")
    assert composed == "a lake :: wallpaper for 2560x1440, stretched"


def test_unknown_lookups_raise(tmp_path):
    conf = cfg.load_config(tmp_path / "nope.yaml")
    with pytest.raises(KeyError):
        conf.profile("ghost")
    with pytest.raises(KeyError):
        conf.model("ghost")
    with pytest.raises(KeyError):
        conf.lora_stack("ghost")
