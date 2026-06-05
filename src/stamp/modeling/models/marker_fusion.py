from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.models.vision_tranformer import _VisionTransformerEncoder


class _MarkerBranchViT(nn.Module):
    """ViT-style encoder for one marker branch."""

    def __init__(
        self,
        *,
        dim_input: int,
        dim_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.encoder = _VisionTransformerEncoder(
            dim_input=dim_input,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi,
        )

    def forward(
        self,
        x: Float[Tensor, "batch token dim_input"],
    ) -> Float[Tensor, "batch dim_model"]:
        batch_size, n_tokens, _ = x.shape
        coords = torch.zeros(batch_size, n_tokens, 2, device=x.device, dtype=x.dtype)
        coords[:, :, 0] = torch.arange(n_tokens, device=x.device, dtype=x.dtype)

        return self.encoder(
            x,
            coords=coords,
            attn_mask=None,
            alibi_mask=None,
        )


class MarkerFusion(nn.Module):
    """
    Marker-aware model with one ViT branch per marker and explicit fusion.

    Expected input formats:
      - (B, M * F): flattened marker features per sample
      - (B, M, F): one feature vector per marker
      - (B, T, M, F): tile-wise marker features, pooled over tiles internally

    The model produces one marker-specific evidence score per output class and
    combines them through an explicit linear fusion layer. This keeps the final
    prediction decomposable into per-marker contributions.
    """

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_markers: int = 7,
        marker_feature_dim: int | None = None,
        dim_model: int = 192,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        use_alibi: bool = False,
    ) -> None:
        super().__init__()

        if n_markers <= 0:
            raise ValueError("n_markers must be positive")

        inferred_marker_dim = marker_feature_dim
        if inferred_marker_dim is None:
            if dim_input % n_markers != 0:
                raise ValueError(
                    "dim_input must be divisible by n_markers when marker_feature_dim "
                    "is not specified"
                )
            inferred_marker_dim = dim_input // n_markers

        self.n_markers = n_markers
        self.marker_feature_dim = inferred_marker_dim
        self.dim_output = dim_output

        self.marker_branches = nn.ModuleList(
            [
                _MarkerBranchViT(
                    dim_input=inferred_marker_dim,
                    dim_model=dim_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_alibi=use_alibi,
                )
                for _ in range(n_markers)
            ]
        )
        self.marker_heads = nn.ModuleList(
            [nn.Linear(dim_model, dim_output) for _ in range(n_markers)]
        )

        # Explicit marker-to-output fusion weights.
        self.fusion_weight = nn.Parameter(torch.ones(n_markers, dim_output))
        self.output_bias = nn.Parameter(torch.zeros(dim_output))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.fusion_weight)
        nn.init.zeros_(self.output_bias)

    def _reshape_inputs(
        self, x: Float[Tensor, "..."]
    ) -> Float[Tensor, "batch marker token feature"]:
        if x.ndim == 4:
            # (B, T, M, F) -> (B, M, T, F)
            x = x.permute(0, 2, 1, 3)

        if x.ndim == 2:
            batch_size, total_dim = x.shape
            expected_dim = self.n_markers * self.marker_feature_dim
            if total_dim != expected_dim:
                raise ValueError(
                    f"Expected flattened marker input of size {expected_dim}, got {total_dim}"
                )
            x = x.view(batch_size, self.n_markers, 1, self.marker_feature_dim)
        elif x.ndim == 3:
            if x.shape[1] != self.n_markers or x.shape[2] != self.marker_feature_dim:
                raise ValueError(
                    "Expected marker input shaped "
                    f"(batch, {self.n_markers}, {self.marker_feature_dim}), got {tuple(x.shape)}"
                )
            x = x.unsqueeze(2)
        elif x.ndim == 4:
            if x.shape[1] != self.n_markers or x.shape[3] != self.marker_feature_dim:
                raise ValueError(
                    "Expected tile-wise marker input shaped "
                    f"(batch, {self.n_markers}, tokens, {self.marker_feature_dim}), got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"Expected a 2D, 3D, or 4D tensor for marker inputs, got {tuple(x.shape)}"
            )

        return x

    @jaxtyped(typechecker=beartype)
    def marker_scores(
        self,
        x: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch marker dim_output"]:
        x = self._reshape_inputs(x)
        marker_scores = []
        for marker_idx, (branch, head) in enumerate(
            zip(self.marker_branches, self.marker_heads, strict=True)
        ):
            marker_tokens = x[:, marker_idx]
            marker_latent = branch(marker_tokens)
            marker_scores.append(head(marker_latent))
        return torch.stack(marker_scores, dim=1)

    @jaxtyped(typechecker=beartype)
    def marker_contributions(
        self,
        x: Float[Tensor, "..."],
    ) -> Float[Tensor, "batch marker dim_output"]:
        scores = self.marker_scores(x)
        return scores * self.fusion_weight.unsqueeze(0)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "..."],
        **kwargs,
    ) -> Float[Tensor, "batch dim_output"]:
        _ = kwargs
        contributions = self.marker_contributions(x)
        return contributions.sum(dim=1) + self.output_bias
