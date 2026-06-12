"""``wallgen`` command-line interface (Typer).

Commands: generate · batch · daily · list-models · loras · train · ui
"""

from __future__ import annotations

import logging
from typing import Optional

import typer

from . import config as cfg
from . import loras as lora_mod
from . import prompts as prompt_mod

app = typer.Typer(add_completion=False, help="Self-hosted FLUX/Z-Image wallpaper generator.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run(conf, prompts, single, profile, loras, lora, seed):
    """Apply lock/compose then generate."""
    from . import pipeline

    composed, overridden = prompt_mod.build_prompts(conf, prompts, single=single)
    if overridden:
        typer.secho(
            f"Prompt lock is '{conf.lock_mode}': ad-hoc prompt ignored; using locked prompts.",
            fg=typer.colors.YELLOW,
        )
    extra = lora_mod.parse_cli_loras(lora)
    records = pipeline.generate(composed, profile, loras, extra, seed, config=conf)
    for r in records:
        typer.echo(r.image_path)
    return records


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for the wallpaper."),
    profile: str = typer.Option(None, "--profile", "-p", help="Profile (dev/prod)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility."),
    loras: Optional[str] = typer.Option(None, "--loras", help="Named LoRA stack from config."),
    lora: list[str] = typer.Option(None, "--lora", help="Ad-hoc LoRA as source:scale (repeatable)."),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a single wallpaper from PROMPT (size + stretch come from config)."""
    _setup_logging(verbose)
    conf = cfg.load_config(config_path)
    _run(conf, [prompt], True, profile, loras, lora, seed)


@app.command()
def batch(
    prompt_file: str = typer.Argument(..., help="File with one prompt per line."),
    profile: str = typer.Option(None, "--profile", "-p"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    loras: Optional[str] = typer.Option(None, "--loras"),
    lora: list[str] = typer.Option(None, "--lora"),
    config_path: Optional[str] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a wallpaper for every prompt in PROMPT_FILE."""
    _setup_logging(verbose)
    conf = cfg.load_config(config_path)
    prompts = prompt_mod.read_prompt_file(prompt_file)
    records = _run(conf, prompts, False, profile, loras, lora, seed)
    typer.echo(f"Generated {len(records)} wallpaper(s).")


@app.command()
def daily(
    config_path: Optional[str] = typer.Option(None, "--config"),
    set_desktop: Optional[bool] = typer.Option(None, "--set-desktop/--no-set-desktop"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the once-a-day scheduled wallpaper job (used by launchd/cron)."""
    _setup_logging(verbose)
    from .scheduler import daily_wallpaper

    typer.echo(daily_wallpaper.run(config_path=config_path, set_desktop=set_desktop))


@app.command("list-models")
def list_models(config_path: Optional[str] = typer.Option(None, "--config")):
    """List configured models and profiles."""
    conf = cfg.load_config(config_path)
    typer.echo("Profiles:")
    for name, p in conf.profiles.items():
        marker = "*" if name == conf.default_profile else " "
        typer.echo(f"  {marker} {name}: model={p.model}")
    typer.echo("Models (priority order):")
    for m in sorted(conf.models.values(), key=lambda m: -m.priority):
        backends = []
        if m.mlx:
            backends.append("mlx")
        if m.base:
            backends.append("diffusers")
        typer.echo(
            f"    {m.name}: kind={m.kind} prio={m.priority} min_ram={m.min_memory_gb}GB "
            f"min_vram={m.min_vram_gb}GB steps={m.default_steps} [{'/'.join(backends)}]"
        )


@app.command("loras")
def loras_cmd(config_path: Optional[str] = typer.Option(None, "--config")):
    """List configured LoRA stacks and local LoRA files."""
    conf = cfg.load_config(config_path)
    typer.echo("Configured stacks:")
    for name, specs in conf.lora_stacks.items():
        typer.echo(f"  {name}: {[(s.source, s.scale) for s in specs]}")
    typer.echo("Local loras/ files:")
    for f in lora_mod.list_local_loras() or ["(none)"]:
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
