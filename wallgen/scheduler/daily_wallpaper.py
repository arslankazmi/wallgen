"""The once-a-day off-hours job: pick today's locked prompt, compose the
wallpaper directive, generate at the prod profile, optionally set the desktop.
Invoked by launchd (macOS) or cron (Linux) — see the templates alongside.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from .. import config as cfg
from .. import loras as lora_mod
from .. import prompts as prompt_mod

log = logging.getLogger("wallgen.scheduler")


def set_desktop_wallpaper(image_path: str | Path) -> bool:
    """Set the OS desktop wallpaper. Returns True on success."""
    image_path = str(Path(image_path).resolve())
    system = platform.system()
    try:
        if system == "Darwin":
            script = f'tell application "System Events" to set picture of every desktop to POSIX file "{image_path}"'
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            return True
        if system == "Linux":
            uri = f"file://{image_path}"
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.background", "picture-uri", uri],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.background", "picture-uri-dark", uri],
                check=False, capture_output=True,
            )
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not set desktop wallpaper: %s", exc)
    return False


def run(config_path: str | None = None, set_desktop: bool | None = None) -> str:
    """Generate today's wallpaper and return its path."""
    from .. import pipeline

    conf = cfg.load_config(config_path)
    sched = conf.scheduler

    # The scheduler always uses locked prompts (deployable, can't be hijacked):
    # today's rotation, composed with the wallpaper directive.
    locked = prompt_mod.read_prompt_file(conf.locked_prompts_path)
    prompt = conf.compose_prompt(prompt_mod.rotate_daily(locked))
    log.info("Daily wallpaper prompt: %r", prompt)

    specs = lora_mod.resolve_stack(conf, None) + cfg.parse_loras(sched.get("loras", []))
    records = pipeline.generate([prompt], profile_name=sched.get("profile", "prod"), extra_loras=specs, config=conf)
    path = records[0].image_path

    want_desktop = sched.get("set_desktop", True) if set_desktop is None else set_desktop
    if want_desktop:
        log.info("Set desktop wallpaper: %s", set_desktop_wallpaper(path))
    return path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(run())
