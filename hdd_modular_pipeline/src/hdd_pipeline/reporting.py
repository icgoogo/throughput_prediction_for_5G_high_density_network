from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

from .utils import memory_mb


class RunReporter:
    """Small JSON run reporter for local/Kaggle experiments."""

    def __init__(self, output_path: str | Path, config: Mapping[str, Any] | None = None):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.report: dict[str, Any] = {
            "status": "running",
            "config": dict(config or {}),
            "events": [],
            "peak_memory_mb": 0.0,
            "started_at": time.time(),
        }
        self.save()

    def event(self, name: str, **extra: Any) -> None:
        mem = memory_mb()
        if mem == mem:
            self.report["peak_memory_mb"] = max(float(self.report["peak_memory_mb"]), mem)
        self.report["events"].append({"name": name, "time": time.time(), "memory_mb": mem, **extra})
        self.save()

    def success(self, metrics: Mapping[str, Any] | None = None) -> None:
        self.report["status"] = "success"
        self.report["metrics"] = dict(metrics or {})
        self.report["finished_at"] = time.time()
        self.save()

    def fail(self, failed_at: str, exc: BaseException) -> None:
        self.report["status"] = "failed"
        self.report["failed_at"] = failed_at
        self.report["error_type"] = type(exc).__name__
        self.report["error"] = str(exc)
        self.report["traceback"] = traceback.format_exc()
        self.report["finished_at"] = time.time()
        self.save()

    def save(self) -> None:
        with self.output_path.open("w") as f:
            json.dump(self.report, f, indent=2, default=str)
