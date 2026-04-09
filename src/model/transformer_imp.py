from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class PeakTransformerImpConfig:
    peak_input_dim: int
    spectrum_input_dim: int
    d_model: int = 160
    num_heads: int = 8
    num_layers: int = 5
    ff_multiplier: float = 4.0
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    output_dim: int = 1
    max_position_embeddings: int = 4096
    local_attention_window: int = 32
    use_global_spectrum_token: bool = True
    use_learned_peak_rank_embedding: bool = True
    use_spectrum_scale_shift: bool = True
    raw_mz_feature_index: int = -1
    use_mz_relative_bias: bool = True
    mz_relative_bias_scale: float = 0.25


class LocalGlobalTransformerEncoderBlock(nn.Module):
    """Transformer block with local peak attention plus a global spectrum token.

    The attention mask keeps the dedicated spectrum token globally connected,
    while peak tokens only attend within a local window in m/z-sorted rank space.
    This is a better match for spectra than unconstrained full attention.
    """

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

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        residual = x
        attn_in = self.norm1(x) if self.use_layer_norm else x
        key_padding_mask: Tensor | None = padding_mask
        if attn_mask is not None and attn_mask.dtype.is_floating_point:
            key_padding_mask = torch.zeros_like(padding_mask, dtype=attn_in.dtype)
            key_padding_mask.masked_fill_(padding_mask, float("-inf"))
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        ff_in = self.norm2(x) if self.use_layer_norm else x
        x = residual + self.dropout(self.ff(ff_in))
        return x


class PeakTransformerImpClassifier(nn.Module):
    """Improved peak denoising transformer.

    Main differences from the baseline model:
    - uses a leaner feature interface by design via config
    - learns rank-based peak positions after sorting by m/z
    - introduces a dedicated spectrum token for global context
    - uses local peak attention plus global-token connectivity
    - conditions peak tokens with spectrum-driven scale/shift before encoding
    - adds an m/z-aware relative attention bias so nearby masses are easier to link
    """

    def __init__(self, config: PeakTransformerImpConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config()

        ff_hidden_dim = int(math.ceil(self.config.d_model * self.config.ff_multiplier))
        self.peak_projection = nn.Linear(self.config.peak_input_dim, self.config.d_model)
        self.spectrum_projection = nn.Linear(
            self.config.spectrum_input_dim, self.config.d_model
        )

        self.peak_token_type = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.spectrum_token_type = nn.Parameter(torch.zeros(1, 1, self.config.d_model))

        self.rank_embedding: nn.Embedding | None = None
        if self.config.use_learned_peak_rank_embedding:
            self.rank_embedding = nn.Embedding(
                self.config.max_position_embeddings,
                self.config.d_model,
            )

        self.spectrum_to_scale: nn.Module | None = None
        self.spectrum_to_shift: nn.Module | None = None
        if self.config.use_spectrum_scale_shift:
            self.spectrum_to_scale = nn.Linear(
                self.config.spectrum_input_dim,
                self.config.d_model,
            )
            self.spectrum_to_shift = nn.Linear(
                self.config.spectrum_input_dim,
                self.config.d_model,
            )

        self.input_norm = nn.LayerNorm(self.config.d_model)
        self.input_dropout = nn.Dropout(self.config.dropout)

        self.encoder = nn.ModuleList(
            [
                LocalGlobalTransformerEncoderBlock(
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
        # The head sees both the refined peak token and the refined spectrum token.
        self.head = nn.Sequential(
            nn.Linear(self.config.d_model * 2, self.config.d_model),
            _get_activation(self.config.activation),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, self.config.output_dim),
        )

        self._hard_attn_mask_cache: dict[tuple[int, torch.device], Tensor] = {}

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

        batch_size, n_peaks, _ = peak_features.shape
        peak_tokens = self.peak_projection(peak_features) + self.peak_token_type
        raw_mz = peak_features[..., self.config.raw_mz_feature_index]

        if self.rank_embedding is not None:
            rank_ids = torch.arange(n_peaks, device=peak_features.device)
            rank_ids = rank_ids.clamp_max(self.config.max_position_embeddings - 1)
            peak_tokens = peak_tokens + self.rank_embedding(rank_ids).unsqueeze(0)

        if self.spectrum_to_scale is not None and self.spectrum_to_shift is not None:
            # Mild FiLM-style conditioning lets global precursor context modulate peaks
            # without overwhelming the peak-local evidence.
            scale = torch.tanh(self.spectrum_to_scale(spectrum_features)).unsqueeze(1)
            shift = self.spectrum_to_shift(spectrum_features).unsqueeze(1)
            peak_tokens = peak_tokens * (1.0 + 0.1 * scale) + shift

        if self.config.use_global_spectrum_token:
            spectrum_token = self.spectrum_projection(spectrum_features).unsqueeze(1)
            spectrum_token = spectrum_token + self.spectrum_token_type
        else:
            # Keep a token-sized zero vector so the rest of the architecture remains
            # shape-stable even if the global token is disabled for ablations.
            spectrum_token = torch.zeros(
                batch_size,
                1,
                self.config.d_model,
                device=peak_features.device,
                dtype=peak_tokens.dtype,
            )

        tokens = torch.cat([spectrum_token, peak_tokens], dim=1)
        tokens = self.input_norm(tokens)
        tokens = self.input_dropout(tokens)

        expanded_padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    device=padding_mask.device,
                    dtype=padding_mask.dtype,
                ),
                padding_mask,
            ],
            dim=1,
        )
        attn_mask = self._build_attention_mask(
            raw_mz=raw_mz,
            padding_mask=padding_mask,
        )

        for block in self.encoder:
            tokens = block(
                tokens,
                padding_mask=expanded_padding_mask,
                attn_mask=attn_mask,
            )

        tokens = self.output_norm(tokens)
        spectrum_summary = tokens[:, :1, :].expand(-1, n_peaks, -1)
        peak_tokens = tokens[:, 1:, :]
        logits = self.head(torch.cat([peak_tokens, spectrum_summary], dim=-1))
        if self.config.output_dim == 1:
            logits = logits.squeeze(-1)

        if logits.ndim == 2:
            logits = logits.masked_fill(padding_mask, 0.0)
        return logits

    def _get_local_global_hard_mask(self, seq_len: int, device: torch.device) -> Tensor:
        cache_key = (seq_len, device)
        cached = self._hard_attn_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        window = self.config.local_attention_window
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # The first token is the global spectrum token: always connected.
        mask[0, :] = False
        mask[:, 0] = False

        # Peak tokens only see a local neighborhood in sorted rank space.
        for idx in range(1, seq_len):
            left = max(1, idx - window)
            right = min(seq_len, idx + window + 1)
            mask[idx, left:right] = False

        self._hard_attn_mask_cache[cache_key] = mask
        return mask

    def _build_attention_mask(
        self,
        raw_mz: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch_size, n_peaks = raw_mz.shape
        seq_len = n_peaks + 1
        device = raw_mz.device

        hard_mask = self._get_local_global_hard_mask(seq_len=seq_len, device=device)
        attn_mask = torch.zeros(
            batch_size,
            seq_len,
            seq_len,
            device=device,
            dtype=raw_mz.dtype,
        )
        attn_mask.masked_fill_(hard_mask.unsqueeze(0), float("-inf"))

        if self.config.use_mz_relative_bias:
            # Use absolute m/z distance as a soft penalty rather than a hard rule.
            # This gives the model a physics-shaped bias without fully preventing
            # long-range communication through the global token.
            mz_delta = torch.abs(raw_mz.unsqueeze(2) - raw_mz.unsqueeze(1))
            mz_bias = -self.config.mz_relative_bias_scale * torch.log1p(mz_delta)
            attn_mask[:, 1:, 1:] = torch.where(
                torch.isinf(attn_mask[:, 1:, 1:]),
                attn_mask[:, 1:, 1:],
                attn_mask[:, 1:, 1:] + mz_bias,
            )

        # MultiheadAttention expects either (L, S) or (B * H, L, S).
        attn_mask = attn_mask.repeat_interleave(self.config.num_heads, dim=0)
        return attn_mask

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
        if self.config.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")
        if self.config.local_attention_window < 1:
            raise ValueError("local_attention_window must be >= 1")
        if not (-self.config.peak_input_dim <= self.config.raw_mz_feature_index < self.config.peak_input_dim):
            raise ValueError("raw_mz_feature_index must point to a valid peak feature")
        if self.config.mz_relative_bias_scale < 0.0:
            raise ValueError("mz_relative_bias_scale must be >= 0.0")
        _get_activation(self.config.activation)

    def _validate_inputs(
        self,
        peak_features: Tensor,
        spectrum_features: Tensor,
        padding_mask: Tensor,
    ) -> None:
        if peak_features.ndim != 3:
            raise ValueError("peak_features must have shape (batch, n_peaks, peak_input_dim)")
        if spectrum_features.ndim != 2:
            raise ValueError("spectrum_features must have shape (batch, spectrum_input_dim)")
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
