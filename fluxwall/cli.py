"""``fluxwall`` command-line interface (Typer).

Commands: generate · batch · daily · list-models · loras · train · ui
"""

from __future__ import annotations

import logging
from typing import Optional

import typer

from . import config as cfg
from . import loras as lora_mod
from . import prompts as prompt_mod

app = typer.Typer(add_completion=False, help="Self-hosted FLUX wallpaper generator.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for the wallpaper."),
    profile: str = typer.Option(None, "--profile", "-p", help="Profile (dev/prod)."),
    resolution: str = typer.Option("hd", "--resolution", "-r", help="Resolution preset."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    loras: Optional[str] = typer.Option(None, "--loras", help="Named LoRA stack from config."),
    lora: list[str] = typer.Option(None, "--lora", help="Ad-hoc LoRA as source:scale (repeatable)."),
    no_upscale: bool = typer.Option(False, "--no-upscale", help="Skip upscaling to target size."),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a single wallpaper from PROMPT."""
    _setup_logging(verbose)
    from . import pipeline  # lazy: avoids importing torch for --help

    conf = cfg.load_config(config_path)
    extra = lora_mod.parse_cli_loras(lora)
    records = pipeline.generate(
        [prompt], profile, resolution, loras, extra, seed,
        do_upscale=not no_upscale, config=conf,
    )
    for r in records:
        typer.echo(r.image_path)


@app.command()
def batch(
    prompt_file: str = typer.Argument(..., help="File with one prompt per line."),
    profile: str = typer.Option(None, "--profile", "-p"),
    resolution: str = typer.Option("hd", "--resolution", "-r"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    loras: Optional[str] = typer.Option(None, "--loras"),
    lora: list[str] = typer.Option(None, "--lora"),
    no_upscale: bool = typer.Option(False, "--no-upscale"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a wallpaper for every prompt in PROMPT_FILE."""
    _setup_logging(verbose)
    from . import pipeline

    conf = cfg.load_config(config_path)
    prompts = prompt_mod.read_prompt_file(prompt_file)
    extra = lora_mod.parse_cli_loras(lora)
    records = pipeline.generate(
        prompts, profile, resolution, loras, extra, seed,
        do_upscale=not no_upscale, config=conf,
    )
    typer.echo(f"Generated {len(records)} wallpaper(s).")
    for r in records:
        typer.echo(r.image_path)


@app.command()
def daily(
    config_path: Optional[str] = typer.Option(None, "--config"),
    set_desktop: Optional[bool] = typer.Option(None, "--set-desktop/--no-set-desktop"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the once-a-day scheduled wallpaper job (used by launchd/cron)."""
    _setup_logging(verbose)
    from .scheduler import daily_wallpaper

    path = daily_wallpaper.run(config_path=config_path, set_desktop=set_desktop)
    typer.echo(path)


@app.command("list-models")
def list_models(config_path: Optional[str] = typer.Option(None, "--config")):
    """List configured models and profiles."""
    conf = cfg.load_config(config_path)
    typer.echo("Profiles:")
    for name, p in conf.profiles.items():
        marker = "*" if name == conf.default_profile else " "
        typer.echo(f"  {marker} {name}: model={p.model} steps={p.steps} guidance={p.guidance}")
    typer.echo("Models:")
    for name, m in conf.models.items():
        gguf = f" gguf={m.gguf}" if m.gguf else ""
        typer.echo(f"    {name}: kind={m.kind} base={m.base}{gguf}")


@app.command("loras")
def loras_cmd(config_path: Optional[str] = typer.Option(None, "--config")):
    """List configured LoRA stacks and local LoRA files."""
    conf = cfg.load_config(config_path)
    typer.echo("Configured stacks:")
    for name, specs in conf.lora_stacks.items():
        typer.echo(f"  {name}: {[ (s.source, s.scale) for s in specs ]}")
    local = lora_mod.list_local_loras()
    typer.echo("Local loras/ files:")
    for f in local or ["(none)"]:
        typer.echo(f"  {f}")


@app.command()
def train(
    dataset: str = typer.Option(..., "--dataset", help="Captioned-image dataset dir."),
    config_name: str = typer.Option("flux_lora_default", "--config", help="Training config template."),
    output: str = typer.Option("loras", "--output", help="Where to write the trained .safetensors."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Train a custom LoRA (delegates to the ak:ml-finetune workflow)."""
    _setup_logging(verbose)
    from .train import train_lora

    train_lora.run(dataset_dir=dataset, config_name=config_name, output_dir=output)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost by default)."),
    port: int = typer.Option(7864, "--port"),
    config_path: Optional[str] = typer.Option(None, "--config"),
):
    """Launch the local Gradio web UI (no external telemetry)."""
    from .webui import launch

    launch(host=host, port=port, config_path=config_path)


if __name__ == "__main__":  # pragma: no cover
    app()
