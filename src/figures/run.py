"""Build every figure and table of a specification, and save the numbers behind them in numbers.json."""

import json
from pathlib import Path

from ..config import Config
from . import prf, qualitative, radar, resolution_views, tables
from .cells import Results
from .spec import FiguresSpec

OUTPUTS = ("prf", "radar", "resolution_views", "tables", "qualitative")


def run_figures(config: Config, spec: FiguresSpec, out_dir: str | Path, only: list[str] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = Results(config, spec)
    wanted = [o for o in (only or OUTPUTS) if getattr(spec, o) is not None]
    numbers = {}

    if "prf" in wanted:
        print("== prf")
        numbers["prf"] = prf.compute(config, spec, results)
        prf.plot(spec, numbers["prf"], out)
    if "radar" in wanted:
        print("== radar")
        values, pooled = radar.radar_values(config, spec, results)
        numbers["radar"] = {"threshold": spec.radar.threshold, "values": values,
                            "objects": {f"{m}/{axis}": objs for (m, axis), objs in pooled.items()}}
        for (m, axis), objs in sorted(pooled.items()):
            print(f"[axes]   {m:12s} {axis:18s} n={len(objs):2d} {' '.join(objs)}")
        radar.plot(spec, values, out)
    if "resolution_views" in wanted:
        print("== resolution_views")
        data = resolution_views.series(spec, results)
        numbers["resolution_views"] = {"object": spec.resolution_views.object, "methods": data}
        resolution_views.plot(spec, data, out)
    if "tables" in wanted:
        print("== tables")
        for name, build in (("coverage", tables.coverage_table), ("chamfer", tables.chamfer_table),
                            ("failures", tables.failures_table)):
            section = getattr(spec.tables, name)
            if section is not None:
                (out / section.output).write_text(build(spec, results))
                print(f"  {section.output}")
    if "qualitative" in wanted:
        print("== qualitative")
        qualitative.crop_renders(config, spec, results, out)
        if spec.qualitative.colorbar is not None:
            qualitative.colorbar(spec, out)

    if numbers:
        path = out / "numbers.json"
        previous = json.loads(path.read_text()) if path.is_file() else {}
        path.write_text(json.dumps({**previous, **numbers}, indent=1))
    return numbers
