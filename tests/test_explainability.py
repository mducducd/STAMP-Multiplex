from pathlib import Path

import h5py
import pandas as pd
import pytest
import torch

from stamp.explainability import explain_slide_
from stamp.modeling.models import LitTileClassifier
from stamp.modeling.models.marker_fusion import MarkerFusion


def _write_marker_feature_file(
    *,
    path: Path,
    marker_embeddings: torch.Tensor,
    marker_names: list[str] | None,
) -> None:
    with h5py.File(path, "w") as h5:
        h5["marker_embeddings"] = marker_embeddings.numpy()
        h5["coords"] = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).numpy()
        h5.attrs["stamp_version"] = "test"
        h5.attrs["extractor"] = "random-test-generator"
        h5.attrs["unit"] = "um"
        h5.attrs["tile_size_um"] = 1
        h5.attrs["tile_size_px"] = 1
        h5.attrs["feat_type"] = "tile"
        if marker_names is not None:
            h5.attrs["marker_names"] = marker_names


def _write_marker_fusion_checkpoint(
    *,
    path: Path,
    n_markers: int,
    marker_feature_dim: int,
    selected_markers: list[str] | None,
) -> None:
    model = LitTileClassifier(
        model_class=MarkerFusion,
        ground_truth_label="target",
        categories=["neg", "pos"],
        category_weights=torch.ones(2),
        dim_input=marker_feature_dim,
        total_steps=1,
        max_lr=1e-3,
        div_factor=25.0,
        train_patients=["train-patient"],
        valid_patients=["valid-patient"],
        model_name="marker_fusion",
        feature_dataset_name="marker_embeddings",
        selected_markers=selected_markers,
        n_markers=n_markers,
        marker_feature_dim=marker_feature_dim,
        dim_model=8,
        n_layers=1,
        n_heads=1,
        dim_feedforward=16,
        dropout=0.0,
        use_alibi=False,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
        },
        path,
    )


def test_explainability_uses_h5_marker_names_for_full_model(tmp_path: Path) -> None:
    h5_path = tmp_path / "slide.h5"
    checkpoint_path = tmp_path / "model.ckpt"
    output_dir = tmp_path / "out"

    _write_marker_feature_file(
        path=h5_path,
        marker_embeddings=torch.randn(2, 3, 4),
        marker_names=["DAPI", "PanCK", "HER2"],
    )
    _write_marker_fusion_checkpoint(
        path=checkpoint_path,
        n_markers=3,
        marker_feature_dim=4,
        selected_markers=None,
    )

    explain_slide_(
        h5_path=h5_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        export_tile_saliency=False,
    )

    marker_df = pd.read_csv(output_dir / "slide" / "marker_summary.csv")
    assert marker_df["marker"].tolist() == ["DAPI", "PanCK", "HER2"]


def test_explainability_uses_checkpoint_selected_marker_order(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "slide.h5"
    checkpoint_path = tmp_path / "model.ckpt"
    output_dir = tmp_path / "out"

    _write_marker_feature_file(
        path=h5_path,
        marker_embeddings=torch.randn(2, 3, 4),
        marker_names=["DAPI", "PanCK", "HER2"],
    )
    _write_marker_fusion_checkpoint(
        path=checkpoint_path,
        n_markers=2,
        marker_feature_dim=4,
        selected_markers=["HER2", "DAPI"],
    )

    explain_slide_(
        h5_path=h5_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        export_tile_saliency=False,
    )

    marker_df = pd.read_csv(output_dir / "slide" / "marker_summary.csv")
    assert marker_df["marker"].tolist() == ["HER2", "DAPI"]


def test_explainability_requires_marker_names_for_subset_checkpoint(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "slide.h5"
    checkpoint_path = tmp_path / "model.ckpt"
    output_dir = tmp_path / "out"

    _write_marker_feature_file(
        path=h5_path,
        marker_embeddings=torch.randn(2, 3, 4),
        marker_names=None,
    )
    _write_marker_fusion_checkpoint(
        path=checkpoint_path,
        n_markers=2,
        marker_feature_dim=4,
        selected_markers=["HER2", "DAPI"],
    )

    with pytest.raises(KeyError, match="checkpoint was trained with marker subset"):
        explain_slide_(
            h5_path=h5_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            export_tile_saliency=False,
        )
