from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class MLPConfig:
    peak_input_dim: int
    spectrum_input_dim: int
    hidden_dims: list[int]
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    output_dim: int = 1
    broadcast_spectrum_features_to_peaks: bool = True


class MLPPeakClassifier(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()

        input_dim = self.config.peak_input_dim + self.config.spectrum_input_dim
        self.network = self._build_mlp(input_dim=input_dim)

    def forward(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
    ) -> Tensor:
        """
        Predict one logit per peak.

        Supported input shapes:

        1) Flattened peaks across spectra:
           peak_features:     (n_peaks_total, peak_input_dim)
           spectrum_features: (n_peaks_total, spectrum_input_dim)
           or
           spectrum_features: (spectrum_input_dim,) for a single spectrum

        2) Batched spectra:
           peak_features:     (batch_size, n_peaks, peak_input_dim)
           spectrum_features: (batch_size, spectrum_input_dim)
           or
           spectrum_features: (batch_size, n_peaks, spectrum_input_dim)

        Returns:
            logits with shape:
            - (n_peaks_total,) for 2D peak input and output_dim == 1
            - (batch_size, n_peaks) for 3D peak input and output_dim == 1
            - otherwise the same leading dimensions with trailing output_dim
        """
        self._validate_inputs(
            peak_features=peak_features,
            spectrum_features=spectrum_features,
        )

        combined_features = self._combine_features(
            peak_features=peak_features,
            spectrum_features=spectrum_features,
        )
        logits = self.network(combined_features)

        if self.config.output_dim == 1:
            logits = logits.squeeze(-1)

        return logits

    def _validate_config(self) -> None:
        if self.config.peak_input_dim <= 0:
            raise ValueError("peak_input_dim must be > 0")
        if self.config.spectrum_input_dim <= 0:
            raise ValueError("spectrum_input_dim must be > 0")
        if not self.config.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if any(dim <= 0 for dim in self.config.hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        if self.config.dropout < 0.0 or self.config.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        allowed_activations = {"relu", "gelu", "silu"}
        if self.config.activation.lower() not in allowed_activations:
            raise ValueError(
                f"activation must be one of {sorted(allowed_activations)}"
            )

    def _validate_inputs(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
    ) -> None:
        if peak_features.ndim not in {2, 3}:
            raise ValueError(
                "peak_features must have shape "
                "(n_peaks_total, peak_input_dim) or "
                "(batch_size, n_peaks, peak_input_dim)"
            )

        if peak_features.shape[-1] != self.config.peak_input_dim:
            raise ValueError(
                f"peak_features last dimension must be {self.config.peak_input_dim}, "
                f"got {peak_features.shape[-1]}"
            )

        if spectrum_features.ndim not in {1, 2, 3}:
            raise ValueError(
                "spectrum_features must have shape "
                "(spectrum_input_dim,), "
                "(n_peaks_total, spectrum_input_dim), "
                "(batch_size, spectrum_input_dim), or "
                "(batch_size, n_peaks, spectrum_input_dim)"
            )

        if spectrum_features.shape[-1] != self.config.spectrum_input_dim:
            raise ValueError(
                f"spectrum_features last dimension must be "
                f"{self.config.spectrum_input_dim}, "
                f"got {spectrum_features.shape[-1]}"
            )

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.config.output_dim))
        return nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        activation = self.config.activation.lower()

        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()

        raise ValueError(f"unsupported activation: {self.config.activation}")

    def _combine_features(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
    ) -> Tensor:
        if peak_features.ndim == 2:
            spectrum_features = self._expand_for_2d_input(
                peak_features=peak_features,
                spectrum_features=spectrum_features,
            )
            return torch.cat([peak_features, spectrum_features], dim=-1)

        spectrum_features = self._expand_for_3d_input(
            peak_features=peak_features,
            spectrum_features=spectrum_features,
        )
        return torch.cat([peak_features, spectrum_features], dim=-1)

    def _expand_for_2d_input(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
    ) -> Tensor:
        n_peaks_total = peak_features.shape[0]

        if spectrum_features.ndim == 1:
            if not self.config.broadcast_spectrum_features_to_peaks:
                raise ValueError(
                    "broadcast_spectrum_features_to_peaks is False, but "
                    "spectrum_features was passed as 1D"
                )
            return spectrum_features.unsqueeze(0).expand(n_peaks_total, -1)

        if spectrum_features.ndim == 2:
            if spectrum_features.shape[0] != n_peaks_total:
                raise ValueError(
                    "For 2D peak_features, 2D spectrum_features must have the same "
                    "number of rows as peak_features"
                )
            return spectrum_features

        raise ValueError("Unsupported spectrum_features shape for 2D peak_features")

    def _expand_for_3d_input(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
    ) -> Tensor:
        batch_size, n_peaks, _ = peak_features.shape

        if spectrum_features.ndim == 2:
            if spectrum_features.shape[0] != batch_size:
                raise ValueError(
                    "For 3D peak_features, 2D spectrum_features must have the same "
                    "batch size as peak_features"
                )
            if not self.config.broadcast_spectrum_features_to_peaks:
                raise ValueError(
                    "broadcast_spectrum_features_to_peaks is False, but "
                    "spectrum_features was passed as 2D"
                )
            return spectrum_features.unsqueeze(1).expand(batch_size, n_peaks, -1)

        if spectrum_features.ndim == 3:
            if spectrum_features.shape[0] != batch_size:
                raise ValueError(
                    "3D spectrum_features batch size must match peak_features"
                )
            if spectrum_features.shape[1] != n_peaks:
                raise ValueError(
                    "3D spectrum_features number of peaks must match peak_features"
                )
            return spectrum_features

        raise ValueError("Unsupported spectrum_features shape for 3D peak_features")