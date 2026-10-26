# Changelog

## Unreleased

### Added
- **`figures` command: a paper's figures and tables from the eval outputs.** The new `src/figures/`
  package builds pooled precision/recall/F-score curves, a radar per surface property, F-score
  against views and resolution, the coverage, Chamfer and failures LaTeX tables, and the crops of the
  qualitative figure. A YAML specification describes the methods (style), the cells that were run,
  and each output. `config/figures/martine_neurips2026.yaml` is the Martine paper's specification,
  provided as an example: the methods' eval outputs are not distributed. A declared cell without
  `metrics.json` is a failure. Radar axes follow the `challenges/*.npy` that are present. See
  [docs/figures.md](docs/figures.md).

### Changed
- **`figures`: size of the PRF figure in the specification** (`prf.figsize`, inches; default 10 x 2.2,
  the former fixed size, output unchanged). The vertical margins are now fixed in inches.
- **Martine figures specification: taller PRF figure, new palette, viridis.** PRF at 10 x 5 in with
  1.0 pt lines. Methods palette: one colour per family; the first member is solid with a filled
  marker, the second dashed with a hollow one; RealityScan now leads the photogrammetry family in the
  legends. The tables keep their order (`tables.failures.methods` is now explicit). Qualitative
  renders and colorbar: viridis instead of cividis.
- **Martine figures specification: RealityScan view subsets.** `config/figures/martine_neurips2026.yaml`
  declares RealityScan at 40, 20, 10 and 5 views (d4), from the RealityScan Linux rerun. The coverage
  table, the failures table and the views/resolution figure now include them.

### Fixed
- **`figures`: an empty reconstruction is a failure.** `evaluate` writes a `metrics.json` with
  `empty_reconstruction: true` (NaN scores) for a mesh with no vertex; `figures` counted that cell as
  reconstructed (coverage table, `n/26`) and pooled its object. It is now a failure like a missing
  `metrics.json`: a cross in the plots, `--` in the Chamfer table, listed in the failures table, and
  left out of every pooled score.
- **Masks are read at the nearest pixel centre for AliceVision cameras.** A projected point was
  looked up in a silhouette by truncation (`x.astype(int)`, i.e. pixel `floor(x)`, kept when
  `0 <= x < w`). AliceVision SfMData cameras — the Martine `eval_cameras.sfm` — put the centre of
  the top-left pixel at (0, 0), so pixel `i` covers `[i - 0.5, i + 0.5)`: every mask was read half
  a pixel off. Each camera dict now carries its convention (`cameras["pixel_centre"]`: `0.0` for
  SfMData; `0.5`, the former truncation, for `cameras.npz`, whose convention is not established) and
  every mask consumer uses it through `visibility.project_to_pixels`: cleanup and GT cleaning
  (`filter_by_visibility`), `visibility_count`, the watertight (hole) culling and the PNG challenge
  sources. **Output change** (SfMData cameras only; `cameras.npz` datasets are unchanged):
  cleanup/GT-cleaning decisions near the dilated silhouettes, the watertight culling of objects with
  `masks_watertight/`, and the per-point `visibility_count`. Ground-truth derivatives computed before
  this fix (e.g. `attributes/visibility_count.npy` of the Martine `gt/` kit) used the truncated
  lookup and must be regenerated to match it.
- **`max_dist` no longer applied to precision/recall/F-score.** `compute_fscore_curve` dropped
  every distance above `evaluation.max_dist` before counting, shrinking only the denominator
  (`n_a`/`n_b`) while the numerator was unaffected — so every reported value was the true value
  divided by coverage (`precision/coverage_a2b`, `recall/coverage_b2a`), a systematic
  overestimate largest exactly where the reconstruction was worst. `max_dist` is always active
  in practice (default `2.0`), so this affected evaluate, recompute, curves and aggregate.
  **Output change:** precision/recall/F-score now decrease (toward their true values) wherever
  coverage is below 1.
- **Reproducible GT point-cloud sampling.** `preprocess_gt` called `downsample_pcd` without
  a seed, so its internal shuffle drew from `np.random.default_rng(None)` (OS entropy) and two
  runs on the same mesh produced `gt_pcd.npy` clouds drifting by ~0.04%. The seed is now
  configurable via `evaluation.sampling_seed` (default `42`, matching the reconstruction-side
  seed in `pipeline/evaluate.py`) and recorded in `gt_pcd_provenance.json`. Already-generated
  clouds are not rewritten; the first regeneration under the default seed will differ by ~0.04%
  from the on-disk cloud (drawn under an unrecorded seed). Expected and harmless.

## Why your earlier numbers may not match

This is the first public release. It is **not** the first working version: the pipeline was
used internally before publication, and four defects fixed in July–August 2026 changed
reported results. If you evaluated with an earlier copy — or built your own harness against
an earlier description of ours — the differences below explain the gap. They are not rounding.

Each entry says what was wrong, why it mattered, and what changed in the output.

---

### 1. Chamfer distance returned `0.0` when no point survived `max_dist`

`compute_chamfer` averaged the distances falling under `max_dist` and fell back to `0.0` when
that set was empty.

An empty set means *not one point of the reconstruction landed within `max_dist` of the ground
truth* — a total failure. And `0.0` is the **best possible** Chamfer distance. So the worst
outcome the pipeline can produce was scored as the best, it sorted first in any ranking, and
it survived every mean taken over objects or methods, dragging aggregates down without trace.

Both directions now yield `NaN` when their kept set is empty, and `NaN` propagates into
`chamfer`. `compute_fscore_curve` does the same when either cloud is empty, filling
precision/recall/F-score with `NaN` and returning early.

This stays distinct from a **legitimate** `0.0`, which is what you get when points do exist
but none fall under threshold *t*. The two were previously indistinguishable.

**Output change:** any cell that reported `chamfer: 0.0` from an empty set now reports `NaN`,
and every aggregate over such cells changes.

---

### 2. `coverage` added

The same bias exists in continuous form: a Chamfer computed on 1% of the points is an
artefact, not a score — and nothing in the output let you detect it. You could not tell an
accurate reconstruction from a tiny fragment that happened to be accurate.

`metrics.json` now carries `n_kept_a2b` / `n_kept_b2a`, `coverage_a2b` / `coverage_b2a`, and
`coverage = min(both)`.

**Read `coverage` before the Chamfer beside it.** High accuracy at 20% coverage describes a
fragment. Previously this had to be re-derived from `n_filtered_*`, if you thought to.

---

### 3. Exclusion masks are applied by one module, identically everywhere

Ground-truth exclusion masks were applied by inline blocks duplicated across the evaluation
and curve paths, each with its own idea of which `.npy` files to honour. The recompute path
skipped them entirely.

The consequence was silent and large: for every object carrying a `challenges/excluded.npy`,
the curves written to disk could be computed on a **different point set** than the
`metrics.json` sitting next to them. Figures and tables derived from the same cell disagreed,
with nothing marking the inconsistency.

`src/core/masking.py` is now the single place that resolves and applies these masks;
`evaluate`, `curves` and `recompute` all route through it, so they cannot diverge again. A
polarity guard refuses any automatic mask that would drop an implausible fraction of the
cloud — it catches an inverted mask, which is otherwise indistinguishable from an aggressive
but valid one, and it fails closed.

The masks actually applied are recorded per cell in `metrics["exclude_masks"]`, so a result
file states what was done rather than what was supposed to be done.

**Output change:** cells that had not received their masks are now evaluated on fewer GT
points.

---

### 4. Dense F-score curves were overwritten by the coarse reporting grid

`recompute` wrote the 5-threshold reporting grid over an existing 100-threshold curve,
destroying the resolution it had been computed at. The file remained valid, so the loss was
invisible; any plot regenerated afterwards was a coarse resampling of itself.

The existing `thresholds.npy` is now read first, and when it is denser than the grid being
written, the curve is recomputed on the existing grid instead of being replaced.

**Output change:** curve-derived quantities are read off the dense grid rather than a 5-point
interpolation of it.

---

### Also fixed

**`gt_pcd.ply` could be selected as the ground-truth mesh.** `get_gt_mesh_path` picked any
`*.ply` containing `gt` and not `clean`. Once GT preprocessing began writing `gt_pcd.ply`
(the resampled point cloud) into the same directory, that file matched the rule. Since
`glob()` returns filesystem order, *which file won was unpredictable* — the same
configuration could evaluate against the mesh on one run and the point cloud on the next,
with no error either way. Derived artefacts are now excluded explicitly and the candidate
list is sorted, so the choice is deterministic.

**Ground truth is sampled from the cleaned mesh.** `preprocess-gt --clean-gt` writes
`gt_cleaned.ply` and resamples `gt_pcd` from it, rather than from the raw merged mesh. This
removes ground-truth points **no camera can see**. For a solid object it changes almost
nothing; for an open, foliage-like object it removes the entire unobservable interior — by
design, since scoring a reconstruction on never-observed surfaces measures nothing. GT point
counts are therefore not comparable across this change; `gt_pcd_provenance.json` records the
source mesh, density and resulting count.

---

## Scope

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score, coverage.
It does not evaluate surface orientation.
