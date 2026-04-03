from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class PeakTransformerConfig:
    peak_input_dim: int
    spectrum_input_dim: int
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ff_multiplier: float = 4.0
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    output_dim: int = 1
    use_spectrum_context_gating: bool = True
    use_peak_positional_projection: bool = True


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            _get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
        )

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        residual = x
        x_attn_in = self.norm1(x) if self.use_layer_norm else x
        attn_out, _ = self.attn(
            x_attn_in,
            x_attn_in,
            x_attn_in,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_ff_in = self.norm2(x) if self.use_layer_norm else x
        x = residual + self.dropout(self.ff(x_ff_in))
        return x


class PeakTransformerClassifier(nn.Module):
    def __init__(self, config: PeakTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()

        ff_hidden_dim = int(math.ceil(self.config.d_model * self.config.ff_multiplier))
        self.peak_projection = nn.Linear(self.config.peak_input_dim, self.config.d_model)
        self.spectrum_projection = nn.Linear(
            self.config.spectrum_input_dim, self.config.d_model
        )
        self.input_norm = nn.LayerNorm(self.config.d_model)
        self.input_dropout = nn.Dropout(self.config.dropout)

        # The first two peak features are m/z and log intensity in the current config.
        # Projecting them separately gives the encoder a lightweight positional/local cue
        # without requiring hard assumptions about a fixed peak count.
        # peak_positional_projection is good for this case, https://medium.com/thedeephub/positional-encoding-explained-a-deep-dive-into-transformer-pe-65cfe8cfe10b
        self.peak_positional_projection: nn.Module | None = None
        if self.config.use_peak_positional_projection:
            self.peak_positional_projection = nn.Linear(2, self.config.d_model)

        self.context_gate: nn.Module | None = None
        if self.config.use_spectrum_context_gating:
            self.context_gate = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.Sigmoid(),
            )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                    use_layer_norm=self.config.use_layer_norm,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            _get_activation(self.config.activation),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, self.config.output_dim),
        )

    def forward(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            peak_features=peak_features,
            spectrum_features=spectrum_features,
            padding_mask=padding_mask,
        )

        peak_tokens = self.peak_projection(peak_features)
        spectrum_context = self.spectrum_projection(spectrum_features).unsqueeze(1)
        tokens = peak_tokens + spectrum_context

        if self.peak_positional_projection is not None:
            tokens = tokens + self.peak_positional_projection(peak_features[..., :2])

        if self.context_gate is not None:
            tokens = tokens * self.context_gate(spectrum_context)

        tokens = self.input_norm(tokens)
        tokens = self.input_dropout(tokens)

        for block in self.encoder:
            tokens = block(tokens, padding_mask=padding_mask)

        tokens = self.output_norm(tokens)
        logits = self.head(tokens)
        if self.config.output_dim == 1:
            logits = logits.squeeze(-1)

        if logits.ndim == 2:
            logits = logits.masked_fill(padding_mask, 0.0)
        return logits

    def _validate_config(self) -> None:
        if self.config.peak_input_dim <= 0:
            raise ValueError("peak_input_dim must be > 0")
        if self.config.spectrum_input_dim <= 0:
            raise ValueError("spectrum_input_dim must be > 0")
        if self.config.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.config.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.config.ff_multiplier <= 1.0:
            raise ValueError("ff_multiplier must be > 1.0")
        if self.config.dropout < 0.0 or self.config.dropout >= 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        _get_activation(self.config.activation)
        if self.config.use_peak_positional_projection and self.config.peak_input_dim < 2:
            raise ValueError(
                "use_peak_positional_projection requires at least two peak features"
            )

    def _validate_inputs(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
        padding_mask: Tensor,
    ) -> None:
        if peak_features.ndim != 3:
            raise ValueError("peak_features must have shape (batch, n_peaks, peak_input_dim)")
        if spectrum_features.ndim != 2:
            raise ValueError(
                "spectrum_features must have shape (batch, spectrum_input_dim)"
            )
        if padding_mask.ndim != 2:
            raise ValueError("padding_mask must have shape (batch, n_peaks)")
        if peak_features.shape[0] != spectrum_features.shape[0]:
            raise ValueError("batch size mismatch between peak_features and spectrum_features")
        if peak_features.shape[:2] != padding_mask.shape:
            raise ValueError("padding_mask must match the first two dims of peak_features")
        if peak_features.shape[-1] != self.config.peak_input_dim:
            raise ValueError(
                f"peak_features last dimension must be {self.config.peak_input_dim}"
            )
        if spectrum_features.shape[-1] != self.config.spectrum_input_dim:
            raise ValueError(
                f"spectrum_features last dimension must be {self.config.spectrum_input_dim}"
            )


def _get_activation(name: str) -> nn.Module:
    activation = name.lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")
