from typer.testing import CliRunner

from fluxwall import pipeline
from fluxwall.cli import app
from fluxwall.organize import GenerationRecord

runner = CliRunner()


def test_list_models_runs():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "flux1-schnell" in result.output
    assert "flux2-klein-4b" in result.output


def test_loras_command_runs():
    result = runner.invoke(app, ["loras"])
    assert result.exit_code == 0
    assert "Configured stacks" in result.output


def test_generate_invokes_pipeline(monkeypatch, tmp_path):
    captured = {}

    def fake_generate(prompts, profile, resolution, lora_stack, extra, seed, do_upscale, config):
        captured.update(
            prompts=prompts, profile=profile, resolution=resolution,
            lora_stack=lora_stack, extra=extra, seed=seed, do_upscale=do_upscale,
        )
        return [GenerationRecord(
            prompt=prompts[0], model="m", profile="dev", device="cpu", dtype="float32",
            quant="none", steps=4, guidance=0.0, seed=seed, gen_size=(1024, 576),
            target_size=(1920, 1080), upscaled=not do_upscale is False,
            image_path=str(tmp_path / "out.png"),
        )]

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    result = runner.invoke(
        app,
        ["generate", "a calm lake", "--profile", "dev", "--resolution", "hd",
         "--seed", "7", "--lora", "org/x:0.6", "--no-upscale"],
    )
    assert result.exit_code == 0, result.output
    assert captured["prompts"] == ["a calm lake"]
    assert captured["profile"] == "dev"
    assert captured["seed"] == 7
    assert captured["do_upscale"] is False
    assert captured["extra"][0].source == "org/x" and captured["extra"][0].scale == 0.6
