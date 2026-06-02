from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ThresholdBundle


def save_thresholds(bundle: ThresholdBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle.to_json(), f, indent=2)


def load_thresholds(path: str | Path) -> ThresholdBundle:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        d: dict[str, Any] = json.load(f)
    return ThresholdBundle.from_json(d)