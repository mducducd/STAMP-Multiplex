from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field


class ExplainabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Field(
        description="Directory to save explainability outputs",
    )
    feature_dir: Path = Field(
        description="Directory containing test-slide feature files",
    )
    checkpoint_path: Path = Field(
        description="Path to the trained model checkpoint",
    )
    feature_paths: list[Path] | None = Field(
        default=None,
        description=(
            "Optional subset of feature files to process. Paths may be relative to "
            "feature_dir or absolute."
        ),
    )
    feature_dataset_name: str = Field(
        default="marker_embeddings",
        description="HDF5 dataset to read from each test-slide feature file.",
    )
    class_index: int | None = Field(
        default=None,
        description=(
            "Optional output class to explain. If omitted for classification, "
            "the predicted class is used."
        ),
    )
    marker_names: list[str] | None = Field(
        default=None,
        description=(
            "Optional override for human-readable marker names. If omitted, "
            "STAMP reads marker names from the HDF5 attribute 'marker_names' "
            "and, for subset-trained models, aligns them to the checkpoint's "
            "selected_markers."
        ),
    )
    export_tile_saliency: bool = Field(
        default=True,
        description="Whether to save tile-by-marker saliency arrays.",
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for explainability computation.",
    )
