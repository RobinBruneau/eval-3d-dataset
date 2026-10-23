"""Click-based CLI for the evaluation pipeline."""

import json
from datetime import datetime
from pathlib import Path

import click

from .config import load_config

# Archived result trees. These are snapshots kept deliberately; sweeping them would
# rewrite history and is almost never what a caller wants.
_BACKUP_MARKERS = ("__submitted_bak", "__pre_cleanup_bak", "__rerun", "__bak")


def _is_backup_path(path: Path) -> bool:
    """True if any path component marks an archived/backup result tree.

    Substring match, not prefix: the markers appear as suffixes in practice
    (`meshroom_d1__rerun2`, `supernormal_d4__bak100k`).
    """
    return any(marker in part for part in path.parts for marker in _BACKUP_MARKERS)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Path to config YAML file")
@click.pass_context
def main(ctx, config):
    """Evaluation pipeline for 3D reconstruction benchmarks."""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    cfg._config_path = Path(config)
    ctx.obj["config"] = cfg
    ctx.obj["config_path"] = config


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--curvature/--no-curvature", default=True, help="Compute curvature")
@click.option("--curvature-radius", type=float, help="Radius for curvature estimation (None = auto)")
@click.option("--visibility/--no-visibility", default=True, help="Compute visibility")
@click.option("--use-masks/--no-masks", "use_masks", default=None, help="Use 2D masks for visibility (None = use config)")
@click.option("--clean-gt/--no-clean-gt", default=True, help="Clean GT mesh by visibility")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.pass_context
def preprocess_gt(ctx, object_name, curvature, curvature_radius, visibility, use_masks, clean_gt, force):
    """Preprocess ground truth: clean mesh, sample point cloud and compute attributes."""
    from .pipeline.preprocess_gt import preprocess_gt as _preprocess_gt

    config = ctx.obj["config"]
    if use_masks is not None:
        config.cleanup.use_masks = use_masks
    if curvature_radius is not None:
        config.evaluation.curvature_radius = curvature_radius
        
    objects = [object_name] if object_name else config.dataset.objects

    for obj in objects:
        _preprocess_gt(config, obj, curvature, visibility, clean_gt, force)


@main.command(name="preprocess-challenges")
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing masks")
@click.pass_context
def preprocess_challenges(ctx, object_name, force):
    """Generate challenge masks from colored PLY meshes and 2D binary masks."""
    from .pipeline.preprocess_challenges import preprocess_challenges as _preprocess

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects

    for obj in objects:
        try:
            _preprocess(config, obj, force)
        except FileNotFoundError as e:
            click.echo(f"Skipping {obj}: {e}")


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option("--use-masks/--no-masks", "use_masks", default=None, help="Use 2D masks (None = use config)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.pass_context
def cleanup(ctx, object_name, method_name, use_masks, force):
    """Clean up meshes by removing non-visible vertices."""
    from .pipeline.cleanup import cleanup_mesh

    config = ctx.obj["config"]
    if use_masks is not None:
        config.cleanup.use_masks = use_masks
        
    objects = [object_name] if object_name else config.dataset.objects
    methods = [method_name] if method_name else config.dataset.methods

    for obj in objects:
        for method in methods:
            method_dir = config.get_method_dir(obj, method)
            lock_path = method_dir / ".cleanup_failed"
            if force and lock_path.exists():
                lock_path.unlink()
            try:
                cleanup_mesh(config, obj, method, force)
            except FileNotFoundError as e:
                click.echo(f"Skipping {obj}/{method}: {e}")
            except Exception as e:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path.write_text(f"{datetime.now().isoformat()}: {e}\n")
                click.echo(f"FAILED {obj}/{method}: {e} (lock: {lock_path})")
                raise


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.pass_context
def evaluate(ctx, object_name, method_name, force):
    """Evaluate meshes: compute distances, Chamfer, F-score."""
    from .pipeline.evaluate import evaluate as _evaluate

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects
    methods = [method_name] if method_name else config.dataset.methods

    for obj in objects:
        for method in methods:
            method_dir = config.get_method_dir(obj, method)
            lock_path = method_dir / ".eval_failed"
            if force and lock_path.exists():
                lock_path.unlink()
            try:
                _evaluate(config, obj, method, force)
            except FileNotFoundError as e:
                click.echo(f"Skipping {obj}/{method}: {e}")
            except Exception as e:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path.write_text(f"{datetime.now().isoformat()}: {e}\n")
                click.echo(f"FAILED {obj}/{method}: {e} (lock: {lock_path})")
                raise


@main.command()
@click.option("--workers", "-w", type=int, help="Max parallel workers")
@click.option("--slurm", is_flag=True, help="Use SLURM instead of local execution")
@click.option("--stages", multiple=True, default=["cleanup", "evaluate"], help="Stages to run")
@click.pass_context
def run(ctx, workers, slurm, stages):
    """Run the full pipeline (cleanup + evaluate) for all objects/methods."""
    config = ctx.obj["config"]
    config_path = ctx.obj["config_path"]

    stages = list(stages)

    if slurm:
        from .runner.slurm import SlurmRunner
        runner = SlurmRunner(config)
    else:
        from .runner.local import LocalRunner
        runner = LocalRunner(config, max_workers=workers)
        runner.config_path = config_path

    results = runner.run_pipeline(config, stages=stages)

    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)
    click.echo(f"\nCompleted: {n_success}/{n_total} jobs succeeded")


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option("--visibility/--no-visibility", default=True, help="Apply visibility masks")
@click.option("--curvature/--no-curvature", default=True, help="Apply curvature masks")
@click.option("--challenges/--no-challenges", default=True, help="Apply challenge masks")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.pass_context
def apply_masks(ctx, object_name, method_name, visibility, curvature, challenges, force):
    """Apply GT attribute masks to extract per-zone distances."""
    from .pipeline.apply_masks import apply_all_masks

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects
    methods = [method_name] if method_name else config.dataset.methods

    for obj in objects:
        for method in methods:
            try:
                apply_all_masks(config, obj, method, visibility, curvature, challenges, force)
            except FileNotFoundError as e:
                click.echo(f"Skipping {obj}/{method}: {e}")


@main.command()
@click.option("--dry-run", is_flag=True, help="Don't actually submit jobs, just show what would be done")
@click.option("--scan-mode", type=click.Choice(["auto", "config"]), default="auto",
              help="Scan mode: 'auto' scans all directories, 'config' uses methods from config")
@click.option("--state-file", type=click.Path(), help="Path to state file (default: beside config)")
@click.pass_context
def watch(ctx, dry_run, scan_mode, state_file):
    """Watch for new meshes and submit evaluation jobs to SLURM.

    Scans results_raw directories for new meshes and submits
    cleanup -> evaluate jobs with SLURM dependencies.
    """
    import logging
    from pathlib import Path
    from .watcher.scanner import run_watcher

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config_path = Path(ctx.obj["config_path"])
    state_path = Path(state_file) if state_file else None

    n_submitted = run_watcher(
        config_path=config_path,
        state_path=state_path,
        dry_run=dry_run,
        scan_mode=scan_mode
    )

    if dry_run:
        click.echo(f"[DRY RUN] Would submit {n_submitted} jobs")
    else:
        click.echo(f"Submitted {n_submitted} jobs")


@main.command()
@click.option("--state-file", type=click.Path(), help="Path to state file (default: beside config)")
@click.pass_context
def watch_status(ctx, state_file):
    """Show status of watched evaluation jobs."""
    from pathlib import Path
    from .watcher.scanner import WatcherState

    config_path = Path(ctx.obj["config_path"])
    state_path = Path(state_file) if state_file else config_path.parent / f".{config_path.stem}_state.json"

    if not state_path.exists():
        click.echo("No state file found. Run 'watch' first.")
        return

    state = WatcherState.load(state_path)

    click.echo(f"Last scan: {state.last_scan or 'Never'}")
    click.echo(f"Total jobs: {len(state.jobs)}")
    click.echo("")

    # Group by status
    by_status = {}
    for job in state.jobs.values():
        by_status.setdefault(job.status, []).append(job)

    for status, jobs in sorted(by_status.items()):
        click.echo(f"[{status.upper()}] ({len(jobs)} jobs)")
        for job in jobs:
            click.echo(f"  - {job.object_name}/{job.method_name}")
            if job.cleanup_job_id:
                click.echo(f"    cleanup: {job.cleanup_job_id}")
            if job.eval_job_id:
                click.echo(f"    eval: {job.eval_job_id}")
            if job.error_message:
                click.echo(f"    error: {job.error_message}")


@main.command()
@click.option("--visibility-groups", type=str, help="JSON: visibility grouping, e.g., '{\"low\": [1, 5], \"high\": [6, 84]}'")
@click.option("--curvature-thresholds", type=str, help="JSON: curvature thresholds, e.g., '{\"concave\": -0.1, \"convex\": 0.1}'")
@click.pass_context
def aggregate(ctx, visibility_groups, curvature_thresholds):
    """Aggregate metrics across all objects with dynamic binning."""
    from .pipeline.aggregate import (
        aggregate_global,
        aggregate_visibility,
        aggregate_curvature,
        aggregate_challenges,
        discover_methods,
        save_aggregated_results,
    )

    config = ctx.obj["config"]

    vis_groups = json.loads(visibility_groups) if visibility_groups else None
    curv_thresh = json.loads(curvature_thresholds) if curvature_thresholds else None

    # `methods: []` in the config means "whatever is evaluated on disk": discover once.
    methods = config.dataset.methods or discover_methods(config)
    if not methods:
        raise click.ClickException("No evaluated method found (no eval_results/metrics.json)")
    click.echo(f"Aggregating {len(methods)} methods over {len(config.dataset.objects)} objects")

    click.echo("Aggregating global metrics...")
    global_metrics = aggregate_global(config, methods=methods)

    click.echo("Aggregating visibility metrics...")
    visibility_metrics = aggregate_visibility(config, methods=methods, grouping=vis_groups)

    click.echo("Aggregating curvature metrics...")
    curvature_metrics = aggregate_curvature(config, methods=methods, thresholds=curv_thresh)

    click.echo("Aggregating challenge metrics...")
    challenge_metrics = aggregate_challenges(config, methods=methods)

    save_aggregated_results(
        config,
        global_metrics=global_metrics,
        visibility_metrics=visibility_metrics,
        curvature_metrics=curvature_metrics,
        challenge_metrics=challenge_metrics,
    )

    output_dir = config.paths.output_root / "aggregated"
    click.echo(f"Saved aggregated results to {output_dir}")


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option("--metrics", default=None,
              help="Comma-separated metrics: uniform,accuracy,completeness,visibility,curvature (default: from config)")
@click.option("--views", default=None,
              help="View indices (e.g., '0,10,20') or 'all'")
@click.option("--scale", default=None, type=float,
              help="Resolution scale factor")
@click.option("--cmap", default=None,
              help="Colormap override (applies to all metrics)")
@click.option("--max-dist", default=None, type=float,
              help="Max distance for accuracy/completeness coloring")
@click.option("--crop/--no-crop", default=None,
              help="Crop to object bounding box")
@click.option("--crop-margin", default=None, type=int,
              help="Margin around bbox when cropping (pixels)")
@click.option("--exclude-mode", default=None, type=click.Choice(["none", "gray", "remove"]),
              help="How to handle excluded regions (default: from config)")
@click.option("--integrated-colorbar/--no-integrated-colorbar", "integrated_colorbar", default=None,
              help="Composite the lateral colorbar onto each view (view_XXX_cb.png) instead of a standalone colorbar.png (default: from config)")
@click.option("--force", "-f", is_flag=True,
              help="Overwrite existing files")
@click.pass_context
def visualize(ctx, object_name, method_name, metrics, views, scale, cmap, max_dist, crop, crop_margin, exclude_mode, integrated_colorbar, force):
    """Generate metric visualization renders."""
    from .core import rendering
    from .pipeline.visualize import visualize_method

    if not rendering.PYRENDER_AVAILABLE:
        # Environment problem, not a per-combo failure: no .viz_failed lock, but a non-zero
        # exit so SLURM marks the job FAILED instead of COMPLETED with nothing rendered.
        raise click.ClickException(
            f"pyrender unavailable ({rendering.PYRENDER_IMPORT_ERROR}). "
            "Install the 'visualization' extra into this venv "
            "(uv pip install -e '.[visualization]') and, headless, set PYOPENGL_PLATFORM=osmesa "
            "with libOSMesa on LD_LIBRARY_PATH (on SLURM: execution.slurm.setup_script, see docs/cluster.md)."
        )

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects
    methods = [method_name] if method_name else config.dataset.methods

    # Parse metrics (use config default if not specified)
    if metrics is None:
        metrics_list = config.visualization.default_metrics
    else:
        metrics_list = [m.strip() for m in metrics.split(",")]

    # Parse views
    view_indices = None
    if views is not None:
        if views.lower() == "all":
            view_indices = None
        else:
            view_indices = [int(v.strip()) for v in views.split(",")]

    # Build cmap overrides
    cmap_overrides = None
    if cmap:
        cmap_overrides = {m: cmap for m in metrics_list}

    for obj in objects:
        for method in methods:
            method_dir = config.get_method_dir(obj, method)
            lock_path = method_dir / ".viz_failed"
            if force and lock_path.exists():
                lock_path.unlink()
            try:
                visualize_method(
                    config,
                    obj,
                    method,
                    metrics=metrics_list,
                    view_indices=view_indices,
                    scale=scale,
                    crop=crop,
                    crop_margin=crop_margin,
                    cmap_overrides=cmap_overrides,
                    max_dist=max_dist,
                    force=force,
                    exclude_mode=exclude_mode,
                    integrated_colorbar=integrated_colorbar,
                )
            except ImportError as e:
                # Missing dependency (e.g. fast_simplification for decimation): an environment
                # error shared by every combo -> fail loudly, but do not lock this combo.
                raise click.ClickException(
                    f"{obj}/{method}: missing dependency ({type(e).__name__}: {e}); "
                    "install the 'visualization' extra into this venv"
                ) from e
            except Exception as e:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path.write_text(f"{datetime.now().isoformat()}: {e}\n")
                click.echo(f"FAILED {obj}/{method}: {e} (lock: {lock_path})")
                raise


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option("--n-thresholds", "-n", default=100, type=int, help="Number of thresholds")
@click.option("--max-threshold", default=1.5, type=float, help="Max threshold in mm")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing curves")
@click.option(
    "--extra-exclude",
    multiple=True,
    help="Additional GT exclusion mask(s), e.g. 'invisible' -> challenges/invisible.npy",
)
@click.pass_context
def curves(ctx, object_name, method_name, n_thresholds, max_threshold, force, extra_exclude):
    """Compute dense precision/recall/F-score curves from saved distances.

    Only runs for combos that already have results_cleaned and distances.
    """
    from .pipeline.curves import compute_curves

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects
    methods = [method_name] if method_name else None
    extra = list(extra_exclude) or None

    computed = 0
    for obj in objects:
        eval_root = config.get_eval_root(obj)
        if not eval_root.exists():
            continue

        if methods:
            method_list = methods
        else:
            method_list = [
                str(d.parent.relative_to(eval_root))
                for d in eval_root.rglob("results_raw")
                if d.is_dir() and "Groundtruth" not in d.parts
            ]

        for method in method_list:
            result = compute_curves(
                config, obj, method, n_thresholds, max_threshold, force, extra_exclude=extra
            )
            if result is not None:
                computed += 1

    click.echo(f"Computed dense curves for {computed} combos")


@main.command()
@click.option("--object", "-o", "object_name", help="Object name (default: all)")
@click.option("--method", "-m", "method_name", help="Method name (default: all)")
@click.option(
    "--extra-exclude",
    multiple=True,
    help="Additional GT exclusion mask(s), e.g. 'invisible' -> challenges/invisible.npy",
)
@click.option(
    "--allow-backup-dirs",
    is_flag=True,
    help="Also process __*_bak / __rerun* directories (off by default: they are archives)",
)
@click.pass_context
def recompute(ctx, object_name, method_name, extra_exclude, allow_backup_dirs):
    """Recompute metrics.json + curves from saved distances (no re-eval needed)."""
    from .pipeline.recompute import recompute_metrics

    config = ctx.obj["config"]
    objects = [object_name] if object_name else config.dataset.objects
    extra = list(extra_exclude) or None

    computed = 0
    for obj in objects:
        eval_root = config.get_eval_root(obj)
        if not eval_root.exists():
            continue

        if method_name:
            method_list = [method_name]
        else:
            method_list = [
                str(d.parent.relative_to(eval_root))
                for d in eval_root.rglob("results_raw")
                if d.is_dir()
                and "Groundtruth" not in d.parts
                and (allow_backup_dirs or not _is_backup_path(d))
            ]

        for method in method_list:
            result = recompute_metrics(config, obj, method, extra_exclude=extra)
            if result is not None:
                computed += 1

    click.echo(f"Recomputed metrics for {computed} combos (max_dist={config.evaluation.max_dist})")


@main.command()
@click.option("--output", "-o", type=click.Path(), default=None, help="Output .tex file")
@click.option("--decimals", default=3, type=int, help="Decimal places for CD values")
@click.option("--no-mean", is_flag=True, help="Skip the Mean row")
@click.pass_context
def latex(ctx, output, decimals, no_mean):
    """Generate LaTeX Chamfer Distance comparison table."""
    from .pipeline.latex_tables import generate_chamfer_table

    config = ctx.obj["config"]

    tex = generate_chamfer_table(config, decimals=decimals, include_mean=not no_mean)

    if output is None:
        out_dir = config.paths.output_root / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "chamfer_table.tex"

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(tex)
    click.echo(f"Written to {output}")


if __name__ == "__main__":
    main()
