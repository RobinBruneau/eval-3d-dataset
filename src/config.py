"""Configuration system using Pydantic for validation."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(description="Dataset name")
    objects: list[str] = Field(default_factory=list, description="List of object names")
    methods: list[str] = Field(default_factory=list, description="List of method names (can include hierarchical paths)")
    exclude_methods: list[str] = Field(default_factory=list, description="Method name substrings to exclude from scanning")
    normals_priors: list[str] = Field(default_factory=list, description="List of normals prior methods")
    num_views: int = Field(default=84, description="Maximum number of camera views")


class PathsConfig(BaseModel):
    """Path configuration with validation.
    
    Structure:
        - data_root: Template path for input data (camera files (cameras.npz, sfm.json, etc.), masks)
          Can contain {object} and {normals_priors} placeholders
          Example: "/path/data/{object}/{normals_priors}"
        - eval_root: Template path for evaluation outputs (GT and method results)
          Can contain {object} placeholder
          Example: "/path/eval/{object}"
          If not specified, defaults to data_root (for backward compatibility)
    """

    data_root: str = Field(description="Root directory template for camera files and masks")
    eval_root: Optional[str] = Field(default=None, description="Eval directory template for GT and method results")
    output_root: Optional[Path] = Field(default=None, description="Output directory for generated scripts and aggregated results")

    @model_validator(mode="after")
    def set_defaults(self) -> "PathsConfig":
        if self.eval_root is None:
            self.eval_root = self.data_root
        if self.output_root is None:
            # Strip {object} and other placeholders to get the base directory
            base = self.eval_root.split("{")[0].rstrip("/")
            self.output_root = Path(base)
        return self


class CleanupConfig(BaseModel):
    """Mesh cleanup configuration."""

    dilation_radius: int = Field(default=12, description="Mask dilation radius in pixels")
    z_threshold: Optional[float] = Field(default=None, description="Remove points below this z")
    use_masks: bool = Field(default=True, description="Use 2D masks for visibility filtering")

    @field_validator("dilation_radius")
    @classmethod
    def validate_dilation_radius(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dilation_radius must be >= 0")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    downsample_density: float = Field(default=0.05, gt=0, description="Point sampling density")
    max_dist: float = Field(default=2.0, gt=0, description="Max distance for filtering outliers")
    curvature_radius: Optional[float] = Field(default=None, description="Radius for curvature estimation (None = auto)")
    fscore_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        description="Thresholds for F-score curve"
    )

    @field_validator("fscore_thresholds")
    @classmethod
    def validate_thresholds(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("fscore_thresholds cannot be empty")
        if any(t <= 0 for t in v):
            raise ValueError("All fscore_thresholds must be positive")
        return v


class SlurmResourceConfig(BaseModel):
    """SLURM resource configuration for a stage."""

    cpus: int = Field(default=4, description="Number of CPUs")
    mem_gb: int = Field(default=32, description="Memory in GB")
    time: str = Field(default="02:00:00", description="Time limit (HH:MM:SS)")


class SlurmConfig(BaseModel):
    """SLURM configuration."""

    account: str = Field(description="SLURM account")
    partition: str = Field(description="SLURM partition")
    cleanup: SlurmResourceConfig = Field(default_factory=lambda: SlurmResourceConfig(cpus=4, mem_gb=48, time="02:00:00"))
    eval: SlurmResourceConfig = Field(default_factory=lambda: SlurmResourceConfig(cpus=16, mem_gb=64, time="04:00:00"))
    viz: SlurmResourceConfig = Field(default_factory=lambda: SlurmResourceConfig(cpus=4, mem_gb=32, time="01:00:00"))


class LocalConfig(BaseModel):
    """Local execution configuration."""

    max_workers: int = Field(default=4, description="Max parallel jobs")


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    mode: str = Field(default="local", description="Execution mode: 'local' or 'slurm'")
    local: LocalConfig = Field(default_factory=LocalConfig)
    slurm: Optional[SlurmConfig] = Field(default=None, description="SLURM config (required if mode='slurm')")

    @model_validator(mode="after")
    def validate_slurm_config(self) -> "ExecutionConfig":
        if self.mode == "slurm" and self.slurm is None:
            raise ValueError("SLURM configuration required when mode='slurm'")
        return self


class AggregationConfig(BaseModel):
    """Aggregation configuration."""

    visibility_groups: Optional[dict[str, list[int]]] = Field(
        default=None,
        description="Visibility grouping, e.g., {'low': [1, 5], 'high': [6, 84]}"
    )
    curvature_thresholds: Optional[dict[str, float]] = Field(
        default=None,
        description="Curvature thresholds, e.g., {'concave': -0.1, 'convex': 0.1}"
    )


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    auto_generate: bool = Field(default=False, description="Generate visualizations after evaluate")
    default_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "completeness"],
        description="Metrics to visualize by default"
    )
    default_views: list[int] | None = Field(
        default=None,
        description="View indices to render (None = all)"
    )
    scale: float = Field(default=1.0, gt=0, description="Resolution scale factor")
    crop: bool = Field(default=False, description="Crop to object bounding box")
    crop_margin: int = Field(default=20, ge=0, description="Margin around bbox when cropping")
    min_component_size: int = Field(default=100, ge=0, description="Min component size for cropping noise removal")
    exclude_mode: str = Field(default="gray", description="How to handle excluded regions: 'none', 'gray', or 'remove'")
    decimation_target: int = Field(default=1_000_000, ge=0, description="Max number of faces for visualization (decimate if exceeded)")
    colormaps: dict[str, str] = Field(
        default_factory=lambda: {
            "accuracy": "jet",
            "completeness": "jet",
            "visibility": "viridis",
            "curvature": "coolwarm",
        },
        description="Default colormaps per metric"
    )
    max_dist: float = Field(default=5.0, gt=0, description="Max distance for error coloring")


class Config(BaseModel):
    """Root configuration model."""

    model_config = {"extra": "allow"}  # Allow extra attributes like _config_path

    dataset: DatasetConfig
    paths: PathsConfig
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    def _resolve_template(self, template: str, object_name: str, method_name: str = None) -> Path:
        """Resolve path template with placeholders.

        Args:
            template: Path template with optional {object}, {normals_priors}, {method} placeholders
            object_name: Object name to substitute
            method_name: Method name (optional)

        Returns:
            Resolved path
        """
        # Extract normals_priors from method name if present AND if it's a known normals_priors
        normals_priors = None
        known_priors = set(self.dataset.normals_priors) if self.dataset.normals_priors else set()

        if method_name and "/" in method_name:
            # Look for a known normals_priors anywhere in the method path
            parts = method_name.split("/")
            for part in parts:
                if part in known_priors:
                    normals_priors = part
                    break

        # Fallback to first normals_priors if none found in method path
        if normals_priors is None and self.dataset.normals_priors:
            normals_priors = self.dataset.normals_priors[0]

        # Resolve template
        path_str = template.format(
            object=object_name,
            normals_priors=normals_priors if normals_priors else "",
            method=method_name if method_name else ""
        )
        return Path(path_str).expanduser().resolve()

    def get_data_root(self, object_name: str, method_name: str = None) -> Path:
        """Get the data root directory for an object (contains cameras.npz, masks).
        
        Args:
            object_name: Object name
            method_name: Method name (used to extract normals_priors from path)
        """
        return self._resolve_template(self.paths.data_root, object_name, method_name)

    def get_eval_root(self, object_name: str) -> Path:
        """Get the eval root directory for an object (contains Groundtruth/ and method results)."""
        return self._resolve_template(self.paths.eval_root, object_name)

    def get_method_dir(self, object_name: str, method_name: str) -> Path:
        """Get the directory for a method result in eval_root."""
        return self.get_eval_root(object_name) / method_name

    def get_gt_dir(self, object_name: str) -> Path:
        """Get the groundtruth directory in eval_root."""
        return self.get_eval_root(object_name) / "Groundtruth"

    def get_masks_issue_watertight_dir(self, object_name: str) -> Path:
        """Get the path to the watertight issue masks directory."""
        return self.get_gt_dir(object_name) / "masks_issue_watertight"

    def get_gt_mesh_path(self, object_name: str, cleaned: bool = False) -> Path:
        """Get the GT mesh path by searching for .ply files in Groundtruth directory.
        
        Args:
            object_name: Object name
            cleaned: If True, return cleaned GT mesh path, else raw GT mesh
        """
        gt_dir = self.get_gt_dir(object_name)
        
        if cleaned:
            # Return path to cleaned GT mesh
            return gt_dir / "gt_cleaned.ply"
        
        if not gt_dir.exists():
            # Return default path for error messages
            return gt_dir / "mesh.ply"
        
        # Search for .ply files.
        # DERIVED products living in the same directory must never be mistaken for the
        # raw GT MESH. 'gt_pcd.ply' (the resampled point cloud) matches the old
        # "contains gt, not clean" rule and would be picked as the source mesh -- with
        # glob() returning filesystem order, that failure would be intermittent and
        # silent. Exclude derived artefacts explicitly, and sort for determinism.
        _DERIVED = ("clean", "pcd", "point", "sampled", "downsample")
        ply_files = sorted(gt_dir.glob("*.ply"))
        if ply_files:
            candidates = [
                f for f in ply_files
                if not any(tag in f.name.lower() for tag in _DERIVED)
            ]
            gt_files = [f for f in candidates if "gt" in f.name.lower()]
            if gt_files:
                return gt_files[0]
            if candidates:
                return candidates[0]
            return ply_files[0]
        
        # Return default path if nothing found
        return gt_dir / "mesh.ply"

    def get_cameras_path(self, object_name: str, method_name: str = None) -> Path:
        """Get the camera file path from data_root.

        Searches for camera files in priority order:
        cameras.npz, sfm.json, cameras.json, then any .sfm file.
        """
        data_root = self.get_data_root(object_name, method_name)

        # Search in priority order
        candidates = [
            data_root / "cameras.npz",
            data_root / "sfm.json",
            data_root / "cameras.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Search for any .sfm file
        sfm_files = list(data_root.glob("*.sfm"))
        if sfm_files:
            return sfm_files[0]

        # Fallback to cameras.npz (will raise FileNotFoundError downstream)
        return data_root / "cameras.npz"

    def get_masks_dir(self, object_name: str, method_name: str = None) -> Path:
        """Get the masks directory from data_root."""
        # Search for "masks" first (AliceVision convention), then "mask"
        masks_dir = self.get_data_root(object_name, method_name) / "masks"
        if masks_dir.exists():
            return masks_dir
        return self.get_data_root(object_name, method_name) / "mask"

    def get_eval_dir(self, object_name: str, method_name: str) -> Path:
        """Get the evaluation output directory."""
        return self.get_method_dir(object_name, method_name) / "eval_results"

    def get_raw_mesh_path(self, object_name: str, method_name: str) -> Path:
        """Get the raw mesh path by searching for .ply or .obj files in results_raw."""
        base_path = self.get_method_dir(object_name, method_name) / "results_raw"
        
        if not base_path.exists():
            # Return default path for error messages
            return base_path / "mesh.ply"
        
        # Search for .ply files first
        ply_files = list(base_path.glob("*.ply"))
        if ply_files:
            return ply_files[0]
        
        # Then search for .obj files
        obj_files = list(base_path.glob("*.obj"))
        if obj_files:
            return obj_files[0]
        
        # Return default path if nothing found
        return base_path / "mesh.ply"

    def get_cleaned_mesh_path(self, object_name: str, method_name: str) -> Path:
        """Get the cleaned mesh path."""
        return self.get_method_dir(object_name, method_name) / "results_cleaned" / "mesh.ply"

    def _resolve_downscale_suffix(self, downscale: str) -> str:
        """Convert downscale name to path suffix: d1->'', d2->'_d2', etc."""
        if downscale == "d1":
            return ""
        return f"_{downscale}"


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Config object.
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    return Config(**raw_config)
