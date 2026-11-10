"""Protocol configuration loader for calendar-shift audit experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> Path:
    repo_candidate = Path(__file__).resolve().parents[2] / "configs" / "protocols.json"
    if repo_candidate.is_file():
        return repo_candidate
    pkg_candidate = Path(__file__).resolve().parent / "protocols.json"
    if pkg_candidate.is_file():
        return pkg_candidate
    raise FileNotFoundError(
        f"Protocol configuration not found at {repo_candidate} or {pkg_candidate}"
    )


def load_protocol_config(path: Optional[Path] = None) -> Dict[str, Any]:
    target = Path(path) if path is not None else default_config_path()
    return json.loads(target.read_text(encoding="utf-8"))
