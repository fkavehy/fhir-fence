"""Configuration loader. Reads configs/*.yaml into a single Settings object so that
no model name, prompt, policy, seed, or scoring parameter is hardcoded in the code.

    from fhirfence.config import load_settings
    cfg = load_settings()                 # finds configs/ automatically
    cfg = load_settings("/path/to/configs")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_configs_dir(explicit: str | os.PathLike | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("FHIRFENCE_CONFIG_DIR")
    if env:
        return Path(env)
    # walk up from this file to the repo root and look for configs/
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / "configs"
        if (cand / "config.yaml").exists():
            return cand
    raise FileNotFoundError(
        "configs/ not found. Pass a path to load_settings() or set FHIRFENCE_CONFIG_DIR."
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Settings:
    """Flat, typed view over the three YAML files."""

    raw: dict[str, Any]          # configs/config.yaml
    prompts: dict[str, Any]      # configs/prompts.yaml
    policy: dict[str, Any]       # configs/policy.yaml
    config_dir: Path

    # ---- convenience accessors (read-only) --------------------------------
    @property
    def global_seed(self) -> int:
        return int(self.raw["reproducibility"]["global_seed"])

    @property
    def canary_seed(self) -> int:
        return int(self.raw["reproducibility"]["canary_seed"])

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw["scoring"]

    @property
    def sensitivity_weights(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["sensitivity"]["classes"].items()}

    @property
    def profile_to_class(self) -> dict[str, str]:
        return dict(self.raw["sensitivity"]["profile_to_class"])

    @property
    def profile_caps(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw["profile_caps"].items()}

    @property
    def default_cap(self) -> int:
        return int(self.raw["default_cap"])

    @property
    def in_scope_profiles(self) -> list[str]:
        return list(self.raw["in_scope_profiles"])

    @property
    def defenses(self) -> list[str]:
        return list(self.raw["defenses"]["order"])

    @property
    def defense_names(self) -> dict[str, str]:
        return dict(self.raw["defenses"]["names"])

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    def api_key(self) -> str:
        """Read the API key from the environment variable named in config.yaml."""
        var = self.model["api_key_env"]
        key = os.environ.get(var)
        if not key:
            raise RuntimeError(
                f"API key not set. Export {var}=... (see .env.example). "
                f"Never commit the key."
            )
        return key


def load_settings(config_dir: str | os.PathLike | None = None) -> Settings:
    cdir = _find_configs_dir(config_dir)
    return Settings(
        raw=_read_yaml(cdir / "config.yaml"),
        prompts=_read_yaml(cdir / "prompts.yaml"),
        policy=_read_yaml(cdir / "policy.yaml"),
        config_dir=cdir,
    )
