# Running the pipeline end to end

[← back to README](../README.md)

```bash
CFG=config/martine.yaml     # or config/skoltech3d.yaml

# 1. Ground truth: clean the mesh, sample it, compute per-point attributes.
#    Writes Groundtruth/{gt_cleaned.ply,gt_pcd.npy,gt_pcd_provenance.json,attributes/}.
#    Skip it for the published Martine dataset: its gt/ kit already ships these files
#    and config/martine.yaml reads them in place (paths.gt_artifacts_root).
eval-pipeline -c $CFG preprocess-gt --clean-gt

# 2. Carve reconstructed vertices falling outside every silhouette.
#    Reads results_raw/mesh.ply, writes results_cleaned/mesh.ply.
eval-pipeline -c $CFG cleanup

# 3. Distances, Chamfer, coverage, F-score at the reporting thresholds.
#    Writes eval_results/metrics.json and eval_results/distances/.
eval-pipeline -c $CFG evaluate

# 4. Dense PR curves from the saved distances (no re-query).
#    Writes eval_results/curves/.
eval-pipeline -c $CFG curves -n 100

# Optional: rebuild metrics.json + curves from saved distances, e.g. after changing
# max_dist or adding a mask. Never re-runs the nearest-neighbour search.
eval-pipeline -c $CFG recompute
```

Scope any stage with `-o/--object` and `-m/--method`; re-run with `-f/--force`.

`preprocess-gt --clean-gt` removes ground-truth points that **no camera sees**, and resamples
from the cleaned surface. For a solid object this changes almost nothing; for an open,
foliage-like object it removes the entire unobservable interior. That is intended — scoring a
reconstruction on surfaces the rig never observed measures nothing — but it means GT point
counts are not comparable across the change. `gt_pcd_provenance.json` records which mesh the
cloud came from, the sampling density and the resulting count.

## Other stages

`eval-pipeline --help` lists every subcommand. Beyond the core flow above:

- `run` — orchestrate cleanup → evaluate → curves over all objects and methods in one call
  (`--slurm` to dispatch on a cluster; see [running on a cluster](cluster.md)).
- `aggregate` — collect per-cell `metrics.json` into summary tables across objects and methods.
- `latex` — emit a LaTeX results table from the aggregated metrics.
- `figures` — build a paper's figures and tables from a YAML specification ([figures.md](figures.md)).
- `visualize` — render per-point metric colourings onto the reconstruction (needs OSMesa; see
  the visualization extra in [install](../README.md#install)).
- `apply-masks`, `preprocess-challenges` — build and apply the optional
  per-point challenge masks described in [configuration](configuration.md#masks).
