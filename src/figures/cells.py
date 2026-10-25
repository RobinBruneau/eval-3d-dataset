"""Per-object results of the declared cells, read from each cell's metrics.json.

Status rule: a declared cell whose metrics.json is absent for an object, or records an empty reconstruction (a mesh
with no vertex: `empty_reconstruction`, written by evaluate), is a failure of the method on that object; an undeclared
cell was never run and is reported as such, never as a failure.
"""

import json
import math
from typing import Optional

from ..config import Config
from .spec import Cell, FiguresSpec

__all__ = ["Cell", "Results"]


class Results:
    def __init__(self, config: Config, spec: FiguresSpec):
        self.config = config
        self.spec = spec
        self.objects = spec.object_list(config)
        self._metrics: dict = {}

    def eval_dir(self, obj: str, cell: Cell):
        return self.config.get_eval_dir(obj, self.spec.cell_dir(cell))

    def metrics(self, obj: str, cell: Cell) -> Optional[dict]:
        """metrics.json of a declared cell, None on failure; KeyError for a cell that was never declared."""
        if not self.spec.is_declared(cell):
            raise KeyError(f"{self.spec.cell_dir(cell)} is not a declared cell")
        key = (obj, self.spec.cell_dir(cell))
        if key not in self._metrics:
            path = self.eval_dir(obj, cell) / "metrics.json"
            m = json.loads(path.read_text()) if path.is_file() else None
            self._metrics[key] = None if m is not None and m.get("empty_reconstruction") else m
        return self._metrics[key]

    def ok(self, obj: str, cell: Cell) -> bool:
        return self.spec.is_declared(cell) and self.metrics(obj, cell) is not None

    def n_ok(self, cell: Cell) -> Optional[int]:
        """Objects reconstructed in this cell, None if it was never run."""
        if not self.spec.is_declared(cell):
            return None
        return sum(self.ok(o, cell) for o in self.objects)

    def failures(self, cell: Cell) -> list[str]:
        if not self.spec.is_declared(cell):
            return []
        return sorted(o for o in self.objects if not self.ok(o, cell))

    def chamfer(self, obj: str, cell: Cell) -> Optional[float]:
        m = self.metrics(obj, cell) if self.spec.is_declared(cell) else None
        v = None if m is None else m.get("chamfer")
        return None if v is None or math.isnan(v) else float(v)

    def fscore_at(self, obj: str, cell: Cell, threshold: float) -> Optional[float]:
        m = self.metrics(obj, cell) if self.spec.is_declared(cell) else None
        if m is None:
            return None
        if threshold not in m["thresholds"]:
            raise ValueError(f"{obj}/{self.spec.cell_dir(cell)}: no F-score at {threshold} in metrics.json "
                             f"(thresholds {m['thresholds']})")
        return float(m["fscore"][m["thresholds"].index(threshold)])
