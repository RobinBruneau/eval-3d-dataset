<div align="center">
<h1>Evaluation Pipeline for Multi-View 3D Surface Reconstruction</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/NeurIPS%202026-Paper-red.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> The official evaluation code for the **[Martine](https://github.com/RobinBruneau/martine_dataset)** benchmark — and a dataset-agnostic pipeline that runs the same metrics on **DiLiGenT-MV**, **LUCES-MV**, **Skoltech3D**, **DTU**, and **EvalMVX**.
</div>

It takes a ground-truth mesh and one reconstructed mesh per method, and produces Chamfer
distance, precision/recall/F-score at fixed thresholds, dense PR curves, a `coverage` figure
telling you how much of the cloud those numbers were actually computed on, and optional metric
visualisations.

It was built for **Martine**, a millimetre-scale 84-view benchmark, but nothing dataset-specific lives in `src/` — any dataset that follows the
[input layout](docs/input-layout.md) works, driven entirely by a config file. Ready-to-edit
configurations for **Martine**, **Skoltech3d**, **DiLiGenT-MV**, **DTU**, **LUCES-MV** and **EvalMVX**
ship in `config/`; see the [datasets guide](docs/datasets.md).

## Table of Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Tests](#tests)
- [Scope](#scope)
- [License](#license)

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[visualization]"   # only for `visualize` (needs OSMesa)
```

Python ≥ 3.10. Dependencies: numpy, trimesh, scikit-learn, OpenCV, pydantic v2, click, PyYAML.
**No GPU is required** for the metrics. The real constraint is memory: nearest-neighbour queries
run over 10–30 M points and take roughly 20 minutes per object×method cell.

Every command takes `-c/--config`; it is required.

```bash
eval-pipeline -c config/martine.yaml --help      # installed entry point
python -m src.cli -c config/martine.yaml --help  # equivalent, without installing
```

## Quick start

```bash
CFG=config/martine.yaml     # or config/skoltech3d.yaml
eval-pipeline -c $CFG preprocess-gt --clean-gt   # clean + sample the GT mesh (Martine: skip, gt/ ships it)
eval-pipeline -c $CFG cleanup                    # carve reconstructions to the silhouettes
eval-pipeline -c $CFG evaluate                   # Chamfer, coverage, F-score
eval-pipeline -c $CFG curves -n 100              # dense PR curves
eval-pipeline -c $CFG figures --spec config/figures/martine_neurips2026.yaml --out figs/  # paper figures
```

Full explanation of each stage: [docs/pipeline.md](docs/pipeline.md).

---

## Documentation

| document | what it covers |
|---|---|
| [docs/input-layout.md](docs/input-layout.md) | The on-disk shape the pipeline expects, GT mesh discovery, camera formats. |
| [docs/configuration.md](docs/configuration.md) | The config keys that change your numbers, masks, and the metre/millimetre units trap. |
| [docs/pipeline.md](docs/pipeline.md) | The end-to-end command flow and every subcommand. |
| [docs/metrics.md](docs/metrics.md) | Every field in `metrics.json`, and why `NaN` is a result. |
| [docs/datasets.md](docs/datasets.md) | The bundled `config/` files, per dataset, and how to add your own. |
| [docs/cluster.md](docs/cluster.md) | Running on SLURM via the job templates. |
| [docs/figures.md](docs/figures.md) | Paper figures and tables (`figures`); the Martine paper's specification as an example. |
| [CHANGELOG.md](CHANGELOG.md) | The four cross-version result changes, in full. |

---

## Tests

```bash
python -m pytest tests/ -q
```

The four `tests/core/test_rendering.py` cases need an OpenGL/OSMesa context and fail in a
headless environment. Everything else must pass.

---

## Scope

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score, coverage.

**Region-restricted evaluation (properties).** By supplying per-point property masks on the
ground truth — `challenges/‹property›.npy` (lambertian, specular, …) — the same metrics are
computed **per zone** instead of over the whole surface. A property counts for an object **if and
only if its `.npy` exists**; a property covering the whole object ships as an all-ones mask. GT
points in `challenges/excluded.npy` (unreliable GT zones) are never evaluated, in any zone. You can therefore score reconstruction quality on precise parts of an
object (a labelled region, thin structures, cavities, …) rather than as a single global number.

**Adding your own challenge.** A *challenge* is a named region of an object that is scored
separately (e.g. `lambertian`, or any name you pick; `excluded` is reserved for the
exclusion mask). The name is **chosen by you and
taken from the file name** — the pipeline discovers challenges from files, so there is nothing to
list by hand. To add one:

1. In your config `.yaml`, point `paths.gt_root` at the read-only folder holding each object's raw
   ground truth; challenge sources live in its `challenges_raw/` sub-folder (see
   [docs/configuration.md](docs/configuration.md) and [docs/input-layout.md](docs/input-layout.md)).
2. Drop a source named after the challenge into `challenges_raw/`, either
   - a copy of the GT mesh with the target region painted **red**, `‹name›.ply`, or
   - per-view binary masks `‹viewId›_‹name›.png`.
3. Run `preprocess-challenges` → it writes `challenges/‹name›.npy`. From there `apply-masks` and the
   aggregation pick every `challenges/*.npy` up **automatically** and report each method per
   challenge under `‹name›` — no code change.

**Not yet implemented.** Normal accuracy — the mean absolute error (MAE) on surface normals — is
planned but not part of the current metrics.

---

## License

This code is released under the **MIT License** — see [`LICENSE`](LICENSE).
Copyright (c) 2026 Baptiste Brument and the Martine authors.

## Citation

If this pipeline supports your research, please cite the **Martine** paper:

```bibtex
@inproceedings{martine2026,
  title={Martine: Benchmarking Multi-View 3D Surface Reconstruction
         Across Viewpoint Coverage, Resolution, and Lighting},
  author={Bruneau, Robin and Brument, Baptiste and Giraud, Frederic and Sigrist, Bastian and Jecklin, Sascha and Fürnstahl, Philipp and Menze, Bjoern and Calvet, Lilian},
  booktitle={NeurIPS},
  year={2026}
}
```
