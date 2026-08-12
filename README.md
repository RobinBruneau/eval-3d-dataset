# eval-3d-dataset

Evaluation pipeline for multi-view 3D surface reconstruction benchmarks.

It takes a ground-truth mesh and one reconstructed mesh per method, and produces Chamfer
distance, precision/recall/F-score at fixed thresholds, dense PR curves, a `coverage` figure
telling you how much of the cloud those numbers were actually computed on, and optional metric
visualisations.

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .

# Not necessary (only for `visualize` (needs OSMesa))
pip install -e ".[visualization]" 
```

Python ≥ 3.10. Dependencies: numpy, trimesh, scikit-learn, OpenCV, pydantic v2, click, PyYAML.
**No GPU is required** for the metrics.

The real constraint is memory: nearest-neighbour queries run over 10–30 M points and take
roughly 20 minutes per object×method cell.

Every command takes `-c/--config`; it is required.
You will have to update the config/martine.yaml file with your paths for `data_root` and `eval_root`. But it should be a path towards the `DATA` and `EVAL` folders from the gdrive. You also have to update `methods` with the names of the method(s) you want to evaluate. Then in `eval_root` next to `Groundtruth`, create a folder(s) according to the same name(s), with inside a folder named `results_raw` and inside your reconstruction as `mesh.ply`. (Look at next section for the architecture details).

In the `DATA` folder, keep the masks folder and sfm.json as in the gdrive and do not remove anything even if you are doing some sparse reconstruction.


---

## Input layout

Both datasets use the same on-disk shape. `{object}` is substituted from `dataset.objects`.

```
<eval_root>/                                 # paths.eval_root, e.g. ./eval/{object}
├── Groundtruth/
│   ├── gt_mesh.ply                          # the GT mesh — you supply this
│   ├── gt_cleaned.ply                       # written by preprocess-gt --clean-gt
│   ├── gt_pcd.npy                           # written by preprocess-gt: (N,3) float
│   ├── gt_pcd_provenance.json               # source mesh, density, N, timestamp
│   ├── attributes/
│   │   ├── curvature_values.npy             # (N,) float32
│   │   └── visibility_count.npy             # (N,) int32 — cameras seeing each point
│   └── challenges/                          # optional (N,) bool masks, True = drop
│       └── excluded.npy
└── <method>/                                # one per entry in dataset.methods
    ├── results_raw/mesh.ply                 # your reconstruction — the only input you supply
    |
    # The following architecture will be generated
    ├── results_cleaned/mesh.ply             # written by cleanup
    └── eval_results/                        # written by evaluate / curves
        ├── metrics.json
        ├── curves/{thresholds,precision,recall,fscore}.npy
        └── distances/{data2gt,gt2data}_{dist,idx}.npy, {data,gt}_points.npy

<data_root>/                                 # paths.data_root, e.g. ./data/{object}/images
├── masks/*.png                              # binary silhouettes, one per view
└── sfm.json                                 # cameras
```

Every `.npy` under `attributes/` and `challenges/` has exactly `N` entries and is indexed on
`gt_pcd.npy` — **not** on the mesh vertices. If you distribute these arrays alongside a GT
mesh, ship `gt_pcd.npy` with them or they index nothing.

**GT mesh discovery.** Any `*.ply` in `Groundtruth/` is a candidate; names containing `clean`,
`pcd`, `point`, `sampled` or `downsample` are excluded as derived products, and the list is
sorted so the choice is deterministic. Name yours `gt_mesh.ply` and there is no ambiguity.

**Cameras.** Searched in order: `cameras.npz`, `sfm.json` (AliceVision SfMData),
`cameras.json`, `*.sfm`. All meshes must already share a frame — the pipeline does not
register.

---

## Run the evaluations: end to end process



```bash

# [STEP 1 ALREADY DONE --> SKIP IT]
# 1. Ground truth: clean the mesh, sample it, compute per-point attributes.
#    Writes Groundtruth/{gt_cleaned.ply,gt_pcd.npy,gt_pcd_provenance.json,attributes/}.
python -m src.cli -c config/martine.yaml preprocess-gt --clean-gt

# 2. Carve reconstructed vertices falling outside every silhouette.
#    Reads results_raw/mesh.ply, writes results_cleaned/mesh.ply.
python -m src.cli -c config/martine.yaml cleanup

# 3. Distances, Chamfer, coverage, F-score at the reporting thresholds.
#    Writes eval_results/metrics.json and eval_results/distances/.
python -m src.cli -c config/martine.yaml evaluate

# 4. Dense PR curves from the saved distances (no re-query).
#    Writes eval_results/curves/.
python -m src.cli -c config/martine.yaml curves -n 100

# Optional: rebuild metrics.json + curves from saved distances, e.g. after changing
# max_dist or adding a mask. Never re-runs the nearest-neighbour search.
python -m src.cli -c config/martine.yaml recompute
```

Scope any stage with `-o/--object` and `-m/--method`; re-run with `-f/--force`.

`preprocess-gt --clean-gt` removes ground-truth points that **no camera sees**, and resamples
from the cleaned surface. For a solid object this changes almost nothing; for an open,
foliage-like object it removes the entire unobservable interior. That is intended — scoring a
reconstruction on surfaces the rig never observed measures nothing — but it means GT point
counts are not comparable across the change. `gt_pcd_provenance.json` records which mesh the
cloud came from, the sampling density and the resulting count.


---

## Configuration

Start from `config/example_martine.yaml` or `config/example_sk3d.yaml`. Both ship with relative
paths (resolved from the working directory); edit the two entries under `paths:` and nothing
else is required.

### The parameters that change your numbers

| key | unit | what it does |
|---|---|---|
| `evaluation.downsample_density` | mesh units | Resampling step for **both** clouds, so a denser mesh cannot buy a better score. Smaller = more points, slower, more memory. |
| `evaluation.max_dist` | mesh units | Distances above this are dropped from the Chamfer average, bounding gross outliers. **Chamfer only** — it does not touch precision/recall/F-score, which are fractions of the full point sets. **Its cost is reported as `coverage`** — if coverage is low, `max_dist` is doing too much work and the Chamfer is not meaningful. |
| `evaluation.fscore_thresholds` | mesh units | The reporting grid. `precision`/`recall`/`fscore` are index-aligned with `thresholds`. Independent of `max_dist`, so a threshold above it is meaningful rather than saturated at `1.0`. |
| `cleanup.dilation_radius` | **pixels** | Silhouette dilation before carving reconstructed vertices that project outside every mask. Larger = more forgiving at the border. `12` for both datasets here. |
| `cleanup.z_threshold` | mesh units | Drops reconstructed points below this height (support plane). `null` disables. |
| `dataset.num_views` | — | Camera views used for masking and visibility. |

### Masks

`Groundtruth/challenges/*.npy` are boolean arrays over the **GT point cloud**, `True` meaning
*drop this point*. They combine with logical OR, and the reconstruction side inherits the
decision through the nearest-neighbour index.

`excluded.npy` is applied automatically whenever the file exists. Whatever was applied is
recorded in `metrics["exclude_masks"]`, so a result file states what was done rather than what
was supposed to be done. Extra masks can be requested per run with `--extra-exclude <name>` on
`curves` and `recompute`.


---

## Reading `metrics.json`

| field | meaning |
|---|---|
| `chamfer_a2b` | mean distance reconstruction → GT over points within `max_dist`. **`NaN` if none survived.** |
| `chamfer_b2a` | mean distance GT → reconstruction, same rule. |
| `chamfer` | mean of both directions; `NaN` propagates. |
| `coverage_a2b` | fraction of reconstruction points within `max_dist` — the share that contributed. |
| `coverage_b2a` | same, GT direction. |
| `coverage` | `min` of the two. **Read this before the Chamfer.** High accuracy at 20% coverage describes a fragment, not a reconstruction. |
| `n_kept_a2b` / `n_kept_b2a` | point counts behind each Chamfer. |
| `n_filtered_a2b` / `n_filtered_b2a` | points dropped by `max_dist`. |
| `thresholds` | reporting grid, echoing `evaluation.fscore_thresholds`. |
| `precision`, `recall`, `fscore` | arrays index-aligned with `thresholds`, computed over the **full** point sets — `max_dist` is not applied here, so a far outlier counts as a miss rather than disappearing from the denominator. All `NaN` if either cloud is empty. |
| `n_gt_points` / `n_data_points` | cloud sizes after resampling and masking. |
| `exclude_masks` | the mask files actually applied, by name. |
| `max_dist` | the value used for the Chamfer clipping, echoed so the file is self-describing. |
| `seed` | resampling seed (42). |

**`NaN` is a result, not a crash.** It means the cell produced nothing measurable — no point
survived `max_dist` for the Chamfer, or a cloud was empty for the curves — and it is
deliberately not `0.0` so it cannot be averaged into a ranking as though it were perfect.

Dense curves live in `eval_results/curves/` as four aligned arrays (`thresholds`, `precision`,
`recall`, `fscore`), independent of the coarse reporting grid.


---

## Other bundled configs

`config/` also carries working configurations for DiLiGenT-MV (`dlmv.yaml`), DTU (`dtu.yaml`),
LUCES-MV (`lucesmv.yaml`) and EvalMVX (`evalmvx.yaml`). These are live configurations rather
than templates and may contain machine-specific paths; use `example_martine.yaml` /
`example_sk3d.yaml` as your starting point.

---

## Scope

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score, coverage.

It does **not** evaluate surface orientation. The normal-map / Mean-Angular-Error path was
removed in August 2026: it had never produced a reported number in any results file, and it was
the only component requiring an unpublished internal library.

---

## Licence

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Baptiste Brument.
