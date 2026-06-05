from collections.abc import Sequence
from typing import TypeAlias

import h5py
import numpy as np

MARKER_EMBEDDINGS_DATASET_NAME = "marker_embeddings"
MarkerSelectionConfig: TypeAlias = str | Sequence[str] | None
NormalizedMarkerSelection: TypeAlias = tuple[str, ...] | None


def normalize_selected_markers(
    selected_markers: MarkerSelectionConfig,
) -> NormalizedMarkerSelection:
    if selected_markers is None:
        return None

    raw_markers = (
        [selected_markers]
        if isinstance(selected_markers, str)
        else list(selected_markers)
    )
    normalized_markers: list[str] = []
    seen_markers: set[str] = set()

    for marker in raw_markers:
        marker_name = marker.strip()
        if not marker_name:
            raise ValueError("selected_markers must not contain empty marker names")
        if marker_name in seen_markers:
            continue
        normalized_markers.append(marker_name)
        seen_markers.add(marker_name)

    if not normalized_markers:
        raise ValueError("selected_markers must contain at least one marker name")

    return tuple(normalized_markers)


def marker_names_from_h5(h5: h5py.File) -> tuple[str, ...] | None:
    marker_names_attr = h5.attrs.get("marker_names")
    if marker_names_attr is None:
        return None

    if isinstance(marker_names_attr, np.ndarray):
        raw_marker_names = marker_names_attr.tolist()
    elif isinstance(marker_names_attr, (list, tuple)):
        raw_marker_names = list(marker_names_attr)
    else:
        raw_marker_names = [marker_names_attr]

    marker_names: list[str] = []
    for marker_name in raw_marker_names:
        if isinstance(marker_name, bytes):
            marker_names.append(marker_name.decode("utf-8"))
        else:
            marker_names.append(str(marker_name))

    return tuple(marker_names)


def resolve_marker_names(
    *,
    available_marker_names: Sequence[str] | None,
    selected_markers: MarkerSelectionConfig,
    expected_count: int,
    context: str,
) -> tuple[str, ...]:
    normalized_markers = normalize_selected_markers(selected_markers)

    if normalized_markers is not None:
        if len(normalized_markers) != expected_count:
            raise ValueError(
                f"{context}: checkpoint expects {expected_count} marker(s), but "
                f"selected_markers resolves to {len(normalized_markers)} marker(s): "
                f"{list(normalized_markers)}"
            )

        if available_marker_names is None:
            raise KeyError(
                f"{context}: checkpoint was trained with marker subset "
                f"{list(normalized_markers)}, but no marker names were available to "
                "map that subset."
            )

        available_marker_names = tuple(
            str(marker_name) for marker_name in available_marker_names
        )
        marker_to_index = {
            marker_name: idx for idx, marker_name in enumerate(available_marker_names)
        }
        missing_markers = [
            marker_name
            for marker_name in normalized_markers
            if marker_name not in marker_to_index
        ]
        if missing_markers:
            raise KeyError(
                f"{context}: checkpoint marker subset {missing_markers} not found in "
                f"available marker names {list(available_marker_names)}"
            )

        return normalized_markers

    if available_marker_names is not None:
        normalized_available = tuple(
            str(marker_name) for marker_name in available_marker_names
        )
        if len(normalized_available) != expected_count:
            raise ValueError(
                f"{context}: expected {expected_count} marker name(s), got "
                f"{len(normalized_available)}: {list(normalized_available)}"
            )
        return normalized_available

    return tuple(f"marker_{i}" for i in range(expected_count))


def resolve_selected_marker_indices(
    *,
    h5: h5py.File,
    dataset_name: str,
    dataset_ndim: int,
    selected_markers: Sequence[str] | None,
) -> np.ndarray | None:
    normalized_markers = normalize_selected_markers(selected_markers)
    if normalized_markers is None:
        return None

    if dataset_ndim < 3:
        raise ValueError(
            f"{h5.filename}: selected_markers requires a marker-aware dataset, "
            f"but '{dataset_name}' has shape rank {dataset_ndim} instead of "
            "at least 3 ([instance, marker, feature])."
        )

    marker_names = marker_names_from_h5(h5)
    if marker_names is None:
        raise KeyError(
            f"{h5.filename}: selected_markers was provided, but the HDF5 attribute "
            "'marker_names' is missing."
        )

    marker_to_index = {marker_name: idx for idx, marker_name in enumerate(marker_names)}
    missing_markers = [
        marker_name
        for marker_name in normalized_markers
        if marker_name not in marker_to_index
    ]
    if missing_markers:
        raise KeyError(
            f"{h5.filename}: selected marker(s) {missing_markers} not found in "
            f"marker_names={list(marker_names)}"
        )

    return np.asarray(
        [marker_to_index[marker_name] for marker_name in normalized_markers],
        dtype=np.int64,
    )


def resolve_marker_loading_config(
    *,
    feature_dataset_name: str,
    selected_markers: MarkerSelectionConfig,
    model_name: str | None,
    context: str,
) -> tuple[str, NormalizedMarkerSelection]:
    normalized_markers = normalize_selected_markers(selected_markers)
    uses_marker_fusion = model_name == "marker_fusion"

    if normalized_markers is not None and model_name not in {None, "marker_fusion"}:
        raise ValueError(
            f"{context}: selected_markers is only supported with "
            "model_name='marker_fusion'."
        )

    resolved_dataset_name = feature_dataset_name
    if (normalized_markers is not None or uses_marker_fusion) and (
        resolved_dataset_name == "auto"
    ):
        resolved_dataset_name = MARKER_EMBEDDINGS_DATASET_NAME

    if (
        normalized_markers is not None
        and resolved_dataset_name != MARKER_EMBEDDINGS_DATASET_NAME
    ):
        raise ValueError(
            f"{context}: selected_markers requires feature_dataset_name to be "
            f"'{MARKER_EMBEDDINGS_DATASET_NAME}' or 'auto', got "
            f"{resolved_dataset_name!r}."
        )

    return resolved_dataset_name, normalized_markers
