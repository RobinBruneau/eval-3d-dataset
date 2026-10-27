"""Specification of a paper's figures and tables: methods, studied cells, style, one section per output.

The specification is a YAML file next to the eval config (example: config/figures/martine_neurips2026.yaml). It
names no path: every input is found through the eval config (eval_root, gt_artifacts_root).
"""

from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class Cell(NamedTuple):
    """One configuration of a method: image resolution tier and number of views (None = all views)."""

    method: str
    tier: str
    views: Optional[int] = None


class MethodSpec(BaseModel):
    display: str
    color: str
    marker: str = "o"
    linestyle: str = "-"
    hollow: bool = Field(default=False, description="Marker drawn with a white face")


class DeclaredCells(BaseModel):
    tiers: list[str] = Field(default_factory=list)
    views: list[int] = Field(default_factory=list, description="View subsets run at CellsSpec.views_tier")


class CellsSpec(BaseModel):
    """The cells of the study. A declared cell without metrics.json is a failure; an undeclared one was never run."""

    tiers: list[str] = Field(description="All resolution tiers, from full to lowest resolution")
    dir: str = "{method}_{tier}"
    views_dir: str = "{method}_{tier}_v{views}"
    views_tier: str = "d4"
    full_views: int = Field(description="Number of views of a plain (all views) cell")
    declared: dict[str, DeclaredCells]


class PRFSpec(BaseModel):
    methods: list[str]
    tier: str
    max_threshold: float = 4.0
    n_thresholds: int = 201
    report: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0], description="Thresholds printed")
    figsize: tuple[float, float] = Field((10, 2.2), description="Figure size (inches): three panels side by side")
    rc: dict = Field(default_factory=dict)
    output: str


class AttributeAxis(BaseModel):
    """Radar axis on a per-point GT attribute: lt (strict) and/or ge/le (closed) bounds."""

    key: str
    attribute: str
    lt: Optional[float] = None
    ge: Optional[float] = None
    le: Optional[float] = None


class RadarSpec(BaseModel):
    methods: list[str]
    tier: str
    threshold: float
    properties: list[str] = Field(description="Property axes, read from the challenges/<property>.npy present")
    attributes: list[AttributeAxis] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    output: str


class ResolutionViewsSpec(BaseModel):
    object: str
    methods: list[str]
    threshold: float = Field(description="F-score threshold, read from metrics.json")
    views: list[int]
    tiers: list[str]
    tier_labels: dict[str, str] = Field(default_factory=dict)
    output: str


class CoverageTable(BaseModel):
    methods: list[str]
    views: list[Union[int, str]] = Field(description="View subsets at the views tier; 'all' = all views")
    tiers: list[str]
    caption: str
    label: str
    output: str


class ChamferTable(BaseModel):
    methods: list[str]
    tiers: list[str]
    method_tiers: dict[str, list[str]] = Field(default_factory=dict,
                                               description="Tiers of a method, in place of `tiers`")
    fscore: Optional[float] = Field(default=None, description="Also report the F-score (%) at this threshold (mm)")
    layout: Literal["cell", "stacked"] = Field(
        default="cell", description="With `fscore`: 'CD / F' in each cell, or a CD block above an F block")
    caption: str
    label: str
    output: str


class FailuresTable(BaseModel):
    methods: Optional[list[str]] = Field(default=None, description="Default: all methods, in spec order")
    caption: str
    label: str
    output: str


class TablesSpec(BaseModel):
    coverage: Optional[CoverageTable] = None
    chamfer: Optional[ChamferTable] = None
    failures: Optional[FailuresTable] = None


class ColorbarSpec(BaseModel):
    vmin: float
    vmax: float
    cmap: str
    label: str
    output: str


class QualitativeSpec(BaseModel):
    """Crops of the error renders made by `eval-pipeline visualize` (<cell>/<render_dir>/<view>)."""

    methods: list[str]
    tier: str
    views: dict[str, str] = Field(description="object -> rendered view file, e.g. view_006.png")
    render_dir: str = "visualizations_remove/accuracy"
    margin: int = 10
    output_dir: str = "qualitative"
    colorbar: Optional[ColorbarSpec] = None


class FiguresSpec(BaseModel):
    objects: Optional[list[str]] = Field(default=None, description="Default: dataset.objects of the eval config")
    object_labels: dict[str, str] = Field(default_factory=dict, description="LaTeX names in the tables")
    methods: dict[str, MethodSpec] = Field(description="Every method, in legend and table order")
    cells: CellsSpec
    style: dict = Field(default_factory=dict, description="matplotlib rcParams applied to every figure")
    prf: Optional[PRFSpec] = None
    radar: Optional[RadarSpec] = None
    resolution_views: Optional[ResolutionViewsSpec] = None
    tables: Optional[TablesSpec] = None
    qualitative: Optional[QualitativeSpec] = None

    @model_validator(mode="after")
    def _known_methods(self):
        used = {"cells.declared": list(self.cells.declared)}
        for name in ("prf", "radar", "resolution_views", "qualitative"):
            section = getattr(self, name)
            if section is not None:
                used[name] = section.methods
        for name in ("coverage", "chamfer", "failures"):
            section = getattr(self.tables, name) if self.tables else None
            if section is not None and section.methods:
                used[f"tables.{name}"] = section.methods
        for where, methods in used.items():
            unknown = [m for m in methods if m not in self.methods]
            if unknown:
                raise ValueError(f"{where}: unknown method(s) {unknown}; declare them under `methods`")
        return self

    def object_list(self, config) -> list[str]:
        return list(self.objects if self.objects is not None else config.dataset.objects)

    def object_label(self, obj: str) -> str:
        return self.object_labels.get(obj, obj.replace("_", "\\_"))

    def _plain(self, cell: Cell) -> Cell:
        return cell._replace(views=None) if cell.views == self.cells.full_views else cell

    def cell_dir(self, cell: Cell) -> str:
        cell = self._plain(cell)
        if cell.views is None:
            return self.cells.dir.format(method=cell.method, tier=cell.tier)
        return self.cells.views_dir.format(method=cell.method, tier=cell.tier, views=cell.views)

    def is_declared(self, cell: Cell) -> bool:
        cell = self._plain(cell)
        d = self.cells.declared.get(cell.method)
        if d is None:
            return False
        if cell.views is None:
            return cell.tier in d.tiers
        return cell.tier == self.cells.views_tier and cell.views in d.views

    def declared_cells(self) -> list[Cell]:
        """Every declared cell: method order, then tier order, a tier's view subsets after it from dense to sparse."""
        out = []
        for m in self.methods:
            d = self.cells.declared.get(m)
            if d is None:
                continue
            for t in self.cells.tiers:
                if t in d.tiers:
                    out.append(Cell(m, t))
                if t == self.cells.views_tier:
                    out += [Cell(m, t, v) for v in sorted(d.views, reverse=True)]
        return out


def load_spec(path: str | Path) -> FiguresSpec:
    with open(path) as f:
        return FiguresSpec(**yaml.safe_load(f))
