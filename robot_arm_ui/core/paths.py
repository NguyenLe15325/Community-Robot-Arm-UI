"""Centralized path resolution for both source and frozen (PyInstaller) environments."""

from __future__ import annotations

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Return the application root directory.

    When running from source this is the repository root.
    When running as a PyInstaller executable this is the directory
    containing the ``.exe``.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    # robot_arm_ui/core/paths.py → three parents up = repo root
    return Path(__file__).resolve().parent.parent.parent


def get_config_dir(subfolder: str = "") -> Path:
    """Return a directory inside ``<base>/config/``, creating it if needed."""
    path = get_base_dir() / "config"
    if subfolder:
        path = path / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_programs_dir() -> Path:
    """Return the top-level ``programs/`` directory, creating it if needed."""
    path = get_base_dir() / "programs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_settings_path() -> Path:
    """Return the default path for ``ui_settings.json``."""
    return get_base_dir() / "config" / "ui_settings.json"
