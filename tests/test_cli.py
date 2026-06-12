from typer.testing import CliRunner

from wallgen import pipeline
from wallgen.cli import app
from wallgen.organize import GenerationRecord

runner = CliRunner()


def test_list_models_runs():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "z-image-turbo" in result.output
    assert "flux2-klein-4b" in result.output
    assert "sd-turbo" in result.output


def test_loras_command_runs():
    result = runner.invoke(app, ["loras"])
    assert result.exit_code == 0
    assert "Configured stacks" in result.output


def _record(prompt, seed, tmp_path):
    return GenerationRecord(
        prompt=prompt, model="z-image-turbo", backend="mlx", profile="dev", device="mps",
        quant="4bit", steps=9, guidance=0.0, seed=seed, gen_size=(1024, 576),
        target_size=(1920, 1080), stretched=True, image_path=str(tmp_path / "out.png"),
    )


def test_generate_invokes_pipeline(monkeypatch, tmp_path):
    captured = {}

    def fake_generate(prompts, profile=None, lora_stack=None, extra_loras=None, seed=None, config=None):
        captured.update(prompts=prompts, profile=profile, extra=extra_loras, seed=seed)
        return [_record(prompts[0], seed, tmp_path)]

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    result = runner.invoke(
        app,
        ["generate", "a calm lake", "--profile", "dev", "--seed", "7", "--lora", "org/x:0.6"],
    )
    assert result.exit_code == 0, result.output
    # Default lock mode is "off" -> prompt passes through unchanged.
    assert captured["prompts"] == ["a calm lake"]
    assert captured["profile"] == "dev"
    assert captured["seed"] == 7
    assert captured["extra"][0].source == "org/x" and captured["extra"][0].scale == 0.6
