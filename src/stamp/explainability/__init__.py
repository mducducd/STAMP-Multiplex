from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from stamp.modeling.data import get_coords
from stamp.modeling.deploy import load_model_from_ckpt
from stamp.modeling.markers import (
    marker_names_from_h5,
    resolve_marker_names,
    resolve_selected_marker_indices,
)

_logger = logging.getLogger("stamp")

__all__ = [
    "ablate_markers_for_slide",
    "explain_slide_",
    "explain_slides_",
    "load_test_slide_inputs",
    "marker_contributions_for_slide",
    "tile_marker_saliency_for_slide",
]


def _resolve_feature_paths(
    *,
    feature_dir: Path,
    feature_paths: Sequence[Path] | None,
) -> list[Path]:
    if feature_paths is None:
        return sorted(feature_dir.glob("*.h5"))

    resolved = []
    for path in feature_paths:
        resolved.append(path if path.is_absolute() else feature_dir / path)
    return resolved


def _get_h5_dataset(h5: h5py.File, dataset_name: str) -> h5py.Dataset:
    if dataset_name not in h5:
        raise KeyError(
            f"{h5.filename}: dataset '{dataset_name}' not found. "
            f"Available datasets: {list(h5.keys())}"
        )
    dataset = h5[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise RuntimeError(
            f"{h5.filename}: expected '{dataset_name}' to be an HDF5 dataset but got {type(dataset)}"
        )
    return dataset


def load_test_slide_inputs(
    *,
    h5_path: Path,
    dataset_name: str = "marker_embeddings",
    selected_markers: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Load one test-slide feature tensor and matching tile coordinates."""
    with h5py.File(h5_path, "r") as h5:
        dataset = _get_h5_dataset(h5, dataset_name)
        feats_np = np.asarray(dataset)
        marker_indices = resolve_selected_marker_indices(
            h5=h5,
            dataset_name=dataset_name,
            dataset_ndim=dataset.ndim,
            selected_markers=selected_markers,
        )
        if marker_indices is not None:
            if feats_np.ndim == 3:
                feats_np = feats_np[:, marker_indices, ...]
            elif feats_np.ndim == 2:
                feats_np = feats_np[marker_indices, ...]
            else:
                raise RuntimeError(
                    f"{h5.filename}: expected marker-aware explainability inputs "
                    f"with rank 2 or 3, got shape {feats_np.shape}"
                )

        feats = torch.from_numpy(feats_np).float()
        coords = torch.from_numpy(get_coords(h5).coords_um).float()

    return feats.to(device), coords.to(device)


def _load_marker_names_from_h5(h5_path: Path) -> list[str] | None:
    with h5py.File(h5_path, "r") as h5:
        marker_names = marker_names_from_h5(h5)
        return list(marker_names) if marker_names is not None else None


def _compute_logits(
    *,
    lit_model: torch.nn.Module,
    feats: Tensor,
    coords: Tensor,
) -> Tensor:
    return cast(
        Tensor,
        lit_model.model(
            feats.unsqueeze(0),
            coords=coords.unsqueeze(0),
            mask=None,
        ).squeeze(0),
    )


def _prediction_summary(
    *,
    lit_model: torch.nn.Module,
    logits: Tensor,
    class_index: int | None,
) -> dict[str, Any]:
    task = lit_model.hparams["task"]

    if task == "classification":
        probabilities = torch.softmax(logits, dim=0)
        selected_class_index = (
            int(probabilities.argmax().item())
            if class_index is None
            else int(class_index)
        )
        categories = list(getattr(lit_model, "categories", []))
        selected_class_name = (
            categories[selected_class_index]
            if categories and selected_class_index < len(categories)
            else str(selected_class_index)
        )
        return {
            "task": task,
            "logits": logits.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "selected_class_index": selected_class_index,
            "selected_class_name": selected_class_name,
            "selected_value": float(probabilities[selected_class_index].item()),
        }

    scalar_value = float(logits.squeeze().item())
    return {
        "task": task,
        "logits": logits.detach().cpu(),
        "probabilities": None,
        "selected_class_index": 0,
        "selected_class_name": task,
        "selected_value": scalar_value,
    }


def _target_scalar(
    *,
    lit_model: torch.nn.Module,
    logits: Tensor,
    class_index: int | None,
) -> tuple[Tensor, int]:
    task = lit_model.hparams["task"]
    if task == "classification":
        target_index = (
            int(logits.argmax().item()) if class_index is None else int(class_index)
        )
        return logits[target_index], target_index
    return logits.squeeze(), 0


def _zero_marker(
    feats: Tensor,
    marker_index: int,
) -> Tensor:
    ablated = feats.clone()
    if ablated.ndim == 3:
        ablated[:, marker_index, :] = 0
    elif ablated.ndim == 2:
        ablated[marker_index, :] = 0
    else:
        raise ValueError(
            f"Expected marker-aware features with 2 or 3 dims, got {tuple(ablated.shape)}"
        )
    return ablated


def marker_contributions_for_slide(
    *,
    checkpoint_path: Path,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:
    lit_model = load_model_from_ckpt(checkpoint_path).eval()
    lit_model = lit_model.to(feats.device)
    return _marker_contributions_for_model(
        lit_model=lit_model,
        feats=feats,
        coords=coords,
        class_index=class_index,
    )


def _marker_contributions_for_model(
    *,
    lit_model: torch.nn.Module,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:

    logits = _compute_logits(lit_model=lit_model, feats=feats, coords=coords)
    summary = _prediction_summary(
        lit_model=lit_model,
        logits=logits,
        class_index=class_index,
    )

    backbone = getattr(lit_model, "model", None)
    if backbone is None or not hasattr(backbone, "marker_scores"):
        raise RuntimeError(
            "Loaded model does not expose marker_scores/marker_contributions"
        )

    selected_index = cast(int, summary["selected_class_index"])
    marker_scores = cast(Tensor, backbone.marker_scores(feats.unsqueeze(0))).squeeze(0)
    marker_contributions = cast(
        Tensor, backbone.marker_contributions(feats.unsqueeze(0))
    ).squeeze(0)

    return {
        **summary,
        "marker_scores": marker_scores.detach().cpu(),
        "marker_contributions": marker_contributions.detach().cpu(),
        "selected_marker_scores": marker_scores[:, selected_index].detach().cpu(),
        "selected_marker_contributions": marker_contributions[:, selected_index]
        .detach()
        .cpu(),
    }


def ablate_markers_for_slide(
    *,
    checkpoint_path: Path,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:
    lit_model = load_model_from_ckpt(checkpoint_path).eval()
    lit_model = lit_model.to(feats.device)
    return _ablate_markers_for_model(
        lit_model=lit_model,
        feats=feats,
        coords=coords,
        class_index=class_index,
    )


def _ablate_markers_for_model(
    *,
    lit_model: torch.nn.Module,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:

    base_logits = _compute_logits(lit_model=lit_model, feats=feats, coords=coords)
    _, target_index = _target_scalar(
        lit_model=lit_model, logits=base_logits, class_index=class_index
    )
    if lit_model.hparams["task"] == "classification":
        baseline = float(torch.softmax(base_logits, dim=0)[target_index].item())
    else:
        baseline = float(base_logits.squeeze().item())

    n_markers = int(getattr(lit_model.model, "n_markers"))
    ablated_values = []
    deltas = []
    for marker_index in range(n_markers):
        ablated_feats = _zero_marker(feats, marker_index)
        ablated_logits = _compute_logits(
            lit_model=lit_model,
            feats=ablated_feats,
            coords=coords,
        )
        if lit_model.hparams["task"] == "classification":
            ablated_value = float(
                torch.softmax(ablated_logits, dim=0)[target_index].item()
            )
        else:
            ablated_value = float(ablated_logits.squeeze().item())
        ablated_values.append(ablated_value)
        deltas.append(baseline - ablated_value)

    return {
        "target_index": target_index,
        "baseline_value": baseline,
        "ablated_values": torch.tensor(ablated_values),
        "delta_from_baseline": torch.tensor(deltas),
    }


def tile_marker_saliency_for_slide(
    *,
    checkpoint_path: Path,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:
    lit_model = load_model_from_ckpt(checkpoint_path).eval()
    lit_model = lit_model.to(feats.device)
    return _tile_marker_saliency_for_model(
        lit_model=lit_model,
        feats=feats,
        coords=coords,
        class_index=class_index,
    )


def _tile_marker_saliency_for_model(
    *,
    lit_model: torch.nn.Module,
    feats: Tensor,
    coords: Tensor,
    class_index: int | None = None,
) -> dict[str, Any]:
    if feats.ndim != 3:
        raise ValueError(
            "Tile-marker saliency expects tile-wise marker features shaped "
            "(tile, marker, feature)"
        )

    tracked_feats = feats.clone().detach().requires_grad_(True)
    logits = _compute_logits(lit_model=lit_model, feats=tracked_feats, coords=coords)
    target_scalar, target_index = _target_scalar(
        lit_model=lit_model,
        logits=logits,
        class_index=class_index,
    )
    target_scalar.backward()

    gradient = tracked_feats.grad
    if gradient is None:
        raise RuntimeError("Could not compute gradients for tile saliency")

    saliency = (tracked_feats * gradient).abs().mean(dim=-1)
    per_tile_total = saliency.sum(dim=-1)

    return {
        "target_index": target_index,
        "tile_marker_saliency": saliency.detach().cpu(),
        "tile_total_saliency": per_tile_total.detach().cpu(),
    }


def _marker_names(
    *,
    count: int,
    marker_names: Sequence[str] | None,
    checkpoint_selected_markers: Sequence[str] | None,
    context: str,
) -> list[str]:
    return list(
        resolve_marker_names(
            available_marker_names=marker_names,
            selected_markers=checkpoint_selected_markers,
            expected_count=count,
            context=context,
        )
    )


def _save_marker_contribution_plot(
    *,
    output_path: Path,
    marker_names: Sequence[str],
    contributions: Tensor,
    selected_class_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    contrib_np = contributions.detach().cpu().numpy()
    colors = ["#c44e52" if v >= 0 else "#4c72b0" for v in contrib_np]
    ax.bar(marker_names, contrib_np, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Contribution")
    ax.set_title(f"Marker Contributions: {selected_class_name}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_tile_saliency_plot(
    *,
    output_path: Path,
    coords: Tensor,
    values: Tensor,
    title: str,
) -> None:
    coords_np = coords.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        coords_np[:, 0],
        coords_np[:, 1],
        c=values_np,
        s=6,
        cmap="inferno",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("coord_x")
    ax.set_ylabel("coord_y")
    ax.invert_yaxis()
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="saliency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_combined_tile_saliency_plot(
    *,
    output_path: Path,
    coords: Tensor,
    total_values: Tensor,
    marker_values: Tensor,
    marker_names: Sequence[str],
) -> None:
    panels: list[tuple[str, Tensor]] = [("Total Tile Saliency", total_values)]
    panels.extend(
        (f"Tile Saliency: {marker_name}", marker_values[:, marker_index])
        for marker_index, marker_name in enumerate(marker_names)
    )

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    coords_np = coords.detach().cpu().numpy()

    for ax, (title, values) in zip(axes_flat, panels, strict=False):
        values_np = values.detach().cpu().numpy()
        scatter = ax.scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            c=values_np,
            s=6,
            cmap="inferno",
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("coord_x")
        ax.set_ylabel("coord_y")
        ax.invert_yaxis()
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="saliency")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def explain_slide_(
    *,
    h5_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_name: str = "marker_embeddings",
    class_index: int | None = None,
    marker_names: Sequence[str] | None = None,
    export_tile_saliency: bool = True,
    device: str | torch.device = "cpu",
) -> None:
    lit_model = load_model_from_ckpt(checkpoint_path).eval()
    checkpoint_selected_markers = cast(
        Sequence[str] | None, lit_model.hparams.get("selected_markers")
    )
    feats, coords = load_test_slide_inputs(
        h5_path=h5_path,
        dataset_name=dataset_name,
        selected_markers=checkpoint_selected_markers,
        device=device,
    )
    lit_model = lit_model.to(feats.device)

    contribution_info = _marker_contributions_for_model(
        lit_model=lit_model,
        feats=feats,
        coords=coords,
        class_index=class_index,
    )
    ablation_info = _ablate_markers_for_model(
        lit_model=lit_model,
        feats=feats,
        coords=coords,
        class_index=class_index,
    )

    names = _marker_names(
        count=len(cast(Tensor, contribution_info["selected_marker_scores"])),
        marker_names=marker_names
        if marker_names is not None
        else _load_marker_names_from_h5(h5_path),
        checkpoint_selected_markers=None
        if marker_names is not None
        else checkpoint_selected_markers,
        context=str(h5_path),
    )

    slide_output_dir = output_dir / h5_path.stem
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    marker_df = pd.DataFrame(
        {
            "marker": names,
            "score": cast(Tensor, contribution_info["selected_marker_scores"]).numpy(),
            "contribution": cast(
                Tensor, contribution_info["selected_marker_contributions"]
            ).numpy(),
            "ablation_delta": cast(
                Tensor, ablation_info["delta_from_baseline"]
            ).numpy(),
            "ablation_value": cast(Tensor, ablation_info["ablated_values"]).numpy(),
        }
    )
    marker_df.to_csv(slide_output_dir / "marker_summary.csv", index=False)

    coords_df = pd.DataFrame(
        coords.detach().cpu().numpy(),
        columns=["coord_x", "coord_y"],
    )
    coords_df.to_csv(slide_output_dir / "tile_coords.csv", index=False)

    summary_payload = {
        "slide": h5_path.name,
        "task": contribution_info["task"],
        "selected_class_index": contribution_info["selected_class_index"],
        "selected_class_name": contribution_info["selected_class_name"],
        "selected_value": contribution_info["selected_value"],
        "dataset_name": dataset_name,
    }
    with open(slide_output_dir / "summary.json", "w") as fp:
        json.dump(summary_payload, fp, indent=2)

    np.save(
        slide_output_dir / "marker_scores.npy",
        cast(Tensor, contribution_info["marker_scores"]).detach().cpu().numpy(),
    )
    np.save(
        slide_output_dir / "marker_contributions.npy",
        cast(Tensor, contribution_info["marker_contributions"]).detach().cpu().numpy(),
    )
    _save_marker_contribution_plot(
        output_path=slide_output_dir / "marker_contribution_barplot.png",
        marker_names=names,
        contributions=cast(Tensor, contribution_info["selected_marker_contributions"]),
        selected_class_name=str(contribution_info["selected_class_name"]),
    )

    if export_tile_saliency:
        saliency_info = _tile_marker_saliency_for_model(
            lit_model=lit_model,
            feats=feats,
            coords=coords,
            class_index=class_index,
        )
        np.save(
            slide_output_dir / "tile_marker_saliency.npy",
            cast(Tensor, saliency_info["tile_marker_saliency"]).detach().cpu().numpy(),
        )
        np.save(
            slide_output_dir / "tile_total_saliency.npy",
            cast(Tensor, saliency_info["tile_total_saliency"]).detach().cpu().numpy(),
        )
        marker_saliency = cast(Tensor, saliency_info["tile_marker_saliency"])
        _save_combined_tile_saliency_plot(
            output_path=slide_output_dir / "tile_saliency_overview.png",
            coords=coords,
            total_values=cast(Tensor, saliency_info["tile_total_saliency"]),
            marker_values=marker_saliency,
            marker_names=names,
        )


def explain_slides_(
    *,
    feature_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    feature_paths: Iterable[Path] | None = None,
    dataset_name: str = "marker_embeddings",
    class_index: int | None = None,
    marker_names: Sequence[str] | None = None,
    export_tile_saliency: bool = True,
    device: str | torch.device = "cpu",
) -> None:
    resolved_paths = _resolve_feature_paths(
        feature_dir=feature_dir,
        feature_paths=list(feature_paths) if feature_paths is not None else None,
    )

    if not resolved_paths:
        raise RuntimeError(f"No feature files found to explain in {feature_dir}")

    cohort_rows = []
    for h5_path in resolved_paths:
        _logger.info("creating explainability outputs for %s", h5_path.name)
        explain_slide_(
            h5_path=h5_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            class_index=class_index,
            marker_names=marker_names,
            export_tile_saliency=export_tile_saliency,
            device=device,
        )
        with open(output_dir / h5_path.stem / "summary.json") as fp:
            cohort_rows.append(json.load(fp))

    if cohort_rows:
        pd.DataFrame(cohort_rows).to_csv(output_dir / "cohort_summary.csv", index=False)
