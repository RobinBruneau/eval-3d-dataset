# Paper figures and tables

`eval-pipeline figures` builds the figures and tables of a paper from the eval outputs. A YAML
specification lists the methods, the configurations that were run and the parameters of each output.

```bash
eval-pipeline -c config/martine.yaml figures --spec config/figures/martine_neurips2026.yaml --out figures_out/
eval-pipeline -c config/martine.yaml figures --spec ... --out figures_out/ --only radar --only tables
```

**The Martine specification is an example.** The reconstructions of the benchmarked methods and their
evaluations are not distributed with the dataset (too heavy). `config/figures/martine_neurips2026.yaml`
is the specification of the Martine paper's figures. To get the same figures, run your methods, evaluate
them with this pipeline, and then point a copy of that file at your own eval outputs.

## Inputs

Everything is found through the eval config. The specification contains no path.

| read from | file | used by |
|---|---|---|
| `<eval_root>/<object>/<cell>/eval_results/` | `metrics.json` | tables, views/resolution |
| same | `distances/{data2gt_dist,gt2data_dist,data2gt_idx}.npy` | PRF curves, radar |
| `<gt_artifacts_root>` (the dataset's `gt/`) | `challenges/<property>.npy`, `challenges/excluded.npy` | radar, every pooled score |
| same | `attributes/{curvature_values,visibility_count}.npy` | radar attribute axes |
| `<eval_root>/<object>/<cell>/<render_dir>/` | renders of `eval-pipeline visualize` | qualitative crops |

A **cell** is one configuration of a method: `<method>_<tier>` (all views) or
`<method>_<tier>_v<views>` (a subset of views at `cells.views_tier`). The directory patterns are
`cells.dir` and `cells.views_dir`.

- A cell listed under `cells.declared` but without `metrics.json` for an object, or whose
  `metrics.json` records an empty reconstruction (`empty_reconstruction`: a mesh with no vertex), is
  a **failure** of the method on that object. It is drawn as a cross, written as `--` in the Chamfer table and listed
  in the failures table. The object is left out of every pooled score.
- A cell that is not declared was **not run**. It is written as `--` in the coverage table and not
  drawn, and it is never counted as a failure.
- The GT points of `challenges/excluded.npy` are removed from every pooled score through
  `core.masking`, as in `evaluate`.

## Outputs

| section | output | what it shows |
|---|---|---|
| `prf` | `prf_curves.pdf` | precision, recall and F-score against the threshold τ, pooled over all objects |
| `radar` | `radar_chart_*.pdf` | F-score at one threshold per surface property and per attribute range |
| `resolution_views` | `resolution_views.pdf` | F-score of one object against the number of views and the image resolution |
| `tables.coverage` | `.tex` | objects reconstructed per method and configuration |
| `tables.chamfer` | `.tex` | Chamfer distance per object and method × resolution (best in bold, second underlined); optionally the F-score at one threshold (`fscore`), as 'CD / F' cells or a second block (`layout`), each metric ranked on its own; `method_tiers` restricts a method's tiers; `mean: pooled` pools the objects' point distances for the Mean row; `sideways` for a landscape page; a list gives several versions |
| `tables.failures` | `.tex` | failed configurations and their objects |
| `qualitative` | `qualitative/<object>_<method>.png`, colorbar PDF | error renders padded for the paper |

`numbers.json` keeps every value behind the figures at full precision: the curves, the radar values
and the objects pooled on each axis, and the views/resolution points.

**Pooling.** The point-level distances of all objects are concatenated before counting, so each
object weighs by its number of points. A point counts when its distance is strictly below the
threshold. The ratios are computed in float64.

**Radar axes.** An object counts on a property axis if and only if its GT has
`challenges/<property>.npy`. An object whose whole surface has the property ships a mask of ones.
Three cases are errors: a mask name that is not a declared property, an object without
`challenges/`, and a missing file that a pool expects. An attribute axis selects GT points with
`lt` (strict upper bound) and/or `ge`/`le` (closed bounds) on `attributes/<attribute>.npy`.

**Qualitative renders** are made beforehand with `visualize`, for each object × method cell. The
command for the Martine paper is in the comment above the `qualitative` section of the
specification.

## Style

`style` holds matplotlib rcParams applied to every figure, and a figure can override them (`prf.rc`).
Each method has a colour, a marker, a line style and `hollow` (white marker face). The Martine
example uses Computer Modern, one hue per family, and a dashed line with hollow markers for the second
method of a family.

## Non-regression

`tests/figures/test_paper_regression.py` rebuilds the Martine paper's outputs and compares them with
the published ones: numbers equal to 1e-9, `.tex` tables identical byte for byte, qualitative crops
identical pixel for pixel. It needs the eval outputs of the methods, so it is skipped unless
`EVAL_FIGURES_REF`, `EVAL_FIGURES_CONFIG` and `EVAL_FIGURES_VIZ_CONFIG` are set (see its docstring).
