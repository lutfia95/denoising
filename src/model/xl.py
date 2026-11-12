"""Mass-aware peak classifier with bounded-memory local and global attention.

Global slots and strong-peak landmarks communicate across the whole spectrum.
Learned pairwise mass biases are computed in bounded query/window chunks.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class XLConfig:
    peak_input_dim: int = 6
    context_input_dim: int = 68
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_multiplier: int = 4
    dropout: float = 0.1
    local_window: int = 64
    full_attention_max_peaks: int = 1024
    global_slots: int = 32
    mz_frequencies: int = 32
    mz_min_wavelength: float = 0.05
    mz_max_wavelength: float = 10000.0
    max_context_tokens: int = 3
    gradient_checkpointing: bool = True
    use_pairwise_mass_bias: bool = True
    pairwise_mass_frequencies: int = 8
    pairwise_mass_hidden_dim: int = 16
    pairwise_query_chunk_size: int = 256
    mass_attention_landmarks: int = 32
    # b/y complements satisfy mz_i + mz_j = M + 2H, a sum the difference bias
    # cannot express. This bias sees (M + 2H - mz_i) - mz_j instead.
    use_complement_bias: bool = False
    complement_neighbors: int = 0
    complement_tolerance_da: float = 0.05

    @classmethod
    def from_checkpoint(cls, values):
        # Earlier XL checkpoints used only local windows and latent global slots.
        values = dict(values)
        values.setdefault("use_pairwise_mass_bias", False)
        values.setdefault("mass_attention_landmarks", 0)
        values.setdefault("use_complement_bias", False)
        values.setdefault("complement_neighbors", 0)
        return cls(**values)


class MassEmbedding(nn.Module):
    """Continuous multi-scale encoding of physical m/z, independent of peak rank."""
    def __init__(self, config):
        super().__init__()
        frequencies = 2 * math.pi / torch.logspace(
            math.log10(config.mz_min_wavelength), math.log10(config.mz_max_wavelength),
            config.mz_frequencies,
        )
        self.register_buffer("frequencies", frequencies)
        self.project = nn.Linear(2 * config.mz_frequencies + 1, config.d_model)

    def forward(self, mz):
        # Phase computation must remain fp32 even under mixed precision.
        phase = mz.float().unsqueeze(-1) * self.frequencies.float()
        encoding = torch.cat([phase.sin(), phase.cos(), mz.float().unsqueeze(-1) / 1000], dim=-1)
        return self.project(encoding)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.q = nn.Linear(config.d_model, config.d_model)
        self.kv = nn.Linear(config.d_model, 2 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def forward(self, query, memory, memory_padding):
        b, n, d = query.shape
        q = self.q(query).view(b, n, self.heads, self.dim).transpose(1, 2)
        kv = self.kv(memory).view(b, memory.shape[1], 2, self.heads, self.dim)
        k, v = kv.unbind(2)
        # SDPA boolean masks use True = attend (opposite to key_padding_mask).
        allowed = ~memory_padding[:, None, None, :]
        value = F.scaled_dot_product_attention(
            q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(value.transpose(1, 2).reshape(b, n, d))


class PairwiseMassBias(nn.Module):
    """Learn a signed, per-head physical mass-difference bias on permitted edges."""
    def __init__(self, config):
        super().__init__()
        frequencies = 2 * math.pi / torch.logspace(
            math.log10(config.mz_min_wavelength), math.log10(config.mz_max_wavelength),
            config.pairwise_mass_frequencies,
        )
        self.register_buffer("frequencies", frequencies)
        self.network = nn.Sequential(
            nn.Linear(2 * config.pairwise_mass_frequencies + 2, config.pairwise_mass_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.pairwise_mass_hidden_dim, config.num_heads, bias=False),
        )
        # Start with ordinary content attention; learn which mass gaps matter.
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, query_mz, key_mz):
        delta = query_mz.float()[:, :, None] - key_mz.float()[:, None, :]
        phase = delta[..., None] * self.frequencies.float()
        features = torch.cat([
            phase.sin(), phase.cos(), delta[..., None] / 1000,
            (delta.sign() * delta.abs().log1p())[..., None] / 10,
        ], -1)
        return self.network(features).permute(0, 3, 1, 2)


class MassAwarePeakAttention(Attention):
    """Full short-spectrum attention or local windows plus sparse remote peaks.

    Long-spectrum windows share a small set of observed, high-intensity landmark
    peaks. These are real peak keys/values, distinct from learned global slots.
    """
    def __init__(self, config):
        super().__init__(config)
        self.window = config.local_window
        self.full_limit = config.full_attention_max_peaks
        self.chunk_size = config.pairwise_query_chunk_size
        self.landmarks = config.mass_attention_landmarks
        self.complement_neighbors = config.complement_neighbors
        self.complement_tolerance_da = config.complement_tolerance_da
        self.checkpoint_attention = config.gradient_checkpointing
        self.mass_bias = PairwiseMassBias(config) if config.use_pairwise_mass_bias else None
        # Same physics encoder, applied to query complement position vs key m/z.
        self.complement_bias = PairwiseMassBias(config) if config.use_complement_bias else None

    @property
    def has_bias(self):
        return self.mass_bias is not None or self.complement_bias is not None

    def _project(self, tokens):
        b, n, _ = tokens.shape
        q = self.q(tokens).view(b, n, self.heads, self.dim).transpose(1, 2)
        k, v = self.kv(tokens).view(b, n, 2, self.heads, self.dim).unbind(2)
        return q, k.transpose(1, 2), v.transpose(1, 2)

    def _attend(self, q, k, v, query_mz, key_mz, key_padding, query_complement=None):
        key_padding = key_padding.clone()
        key_padding[key_padding.all(-1), 0] = False
        if not self.has_bias:
            mask = ~key_padding[:, None, None, :]
        else:
            bias = 0
            if self.mass_bias is not None:
                bias = self.mass_bias(query_mz, key_mz)
            if self.complement_bias is not None:
                bias = bias + self.complement_bias(query_complement, key_mz)
            mask = bias.to(q.dtype).masked_fill(key_padding[:, None, None, :], float("-inf"))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )

    def _run_attention(self, *inputs):
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._attend, *inputs, use_reentrant=False)
        return self._attend(*inputs)

    def _full(self, peaks, mz, padding, complement=None):
        if not self.has_bias:
            return super().forward(peaks, peaks, padding)
        q, k, v = self._project(peaks)
        pieces = []
        complement_chunks = (complement.split(self.chunk_size, dim=1) if complement is not None
                             else [None] * len(mz.split(self.chunk_size, dim=1)))
        for q_chunk, mz_chunk, c_chunk in zip(q.split(self.chunk_size, dim=2), mz.split(self.chunk_size, dim=1),
                                              complement_chunks):
            pieces.append(self._run_attention(q_chunk, k, v, mz_chunk, mz, padding, c_chunk))
        value = torch.cat(pieces, dim=2).transpose(1, 2).reshape_as(peaks)
        return self.out(value)

    def _complement_indices(self, mz, padding, complement):
        """Bounded nearest partners in sorted m/z; -1 denotes no valid edge."""
        count = self.complement_neighbors
        searchable = mz.float().masked_fill(padding, float("inf")).contiguous()
        positions = torch.searchsorted(searchable, complement.float().contiguous())
        offsets = torch.arange(-count, count + 1, device=mz.device)
        candidates = positions[..., None] + offsets
        safe = candidates.clamp(0, mz.shape[1] - 1)
        owners = torch.arange(len(mz), device=mz.device)[:, None, None]
        distances = (searchable[owners, safe] - complement[..., None]).abs()
        invalid = ((candidates < 0) | (candidates >= mz.shape[1]) | padding[..., None]
                   | (candidates == torch.arange(mz.shape[1], device=mz.device)[None, :, None]))
        distances = distances.masked_fill(invalid, float("inf"))
        # Stable ordering makes equal-distance candidates reproducible.
        selected = distances.argsort(dim=-1, stable=True)[..., :count]
        indices = candidates.gather(-1, selected)
        return indices.masked_fill(distances.gather(-1, selected) > self.complement_tolerance_da, -1)

    def _windows(self, peaks, mz, padding, intensity, shift, complement=None):
        b, n, d = peaks.shape
        right = (-n - shift) % self.window
        tokens = F.pad(peaks, (0, 0, shift, right))
        padded_length = tokens.shape[1]
        windows = padded_length // self.window
        window_padding = F.pad(padding, (shift, right), value=True).reshape(b * windows, self.window)
        if not self.has_bias and not self.landmarks and not self.complement_neighbors:
            # Preserve the original XL attention path for old checkpoints.
            safe_padding = window_padding.clone()
            safe_padding[safe_padding.all(-1), 0] = False
            local = tokens.reshape(b * windows, self.window, d)
            update = super().forward(local, local, safe_padding).reshape(b, padded_length, d)
            return update[:, shift:shift + n]

        q, k, v = self._project(tokens)
        full_k, full_v = k.transpose(1, 2), v.transpose(1, 2)
        neighbors = None
        if self.complement_neighbors:
            neighbors = self._complement_indices(mz, padding, complement)
            neighbors = F.pad(neighbors, (0, 0, shift, right), value=-1).reshape(
                b * windows, self.window * self.complement_neighbors)

        def window_view(value):
            return value.reshape(b, self.heads, windows, self.window, self.dim).permute(0, 2, 1, 3, 4).reshape(
                b * windows, self.heads, self.window, self.dim)

        q, k, v = map(window_view, (q, k, v))
        window_mz = F.pad(mz, (shift, right)).reshape(b * windows, self.window)
        window_complement = (F.pad(complement, (shift, right)).reshape(b * windows, self.window)
                             if complement is not None else None)
        count = min(self.landmarks, n)
        # Stable ties preserve predictions when padding or batch lengths change.
        anchor_indices = intensity.masked_fill(padding, float("-inf")).argsort(descending=True, stable=True)[:, :count]
        anchors = peaks.gather(1, anchor_indices[..., None].expand(-1, -1, d))
        _, anchor_k, anchor_v = self._project(anchors)
        anchor_mz = mz.gather(1, anchor_indices)
        anchor_padding = padding.gather(1, anchor_indices)
        per_chunk = max(1, self.chunk_size // self.window)
        pieces, offset = [], 0
        complement_chunks = (window_complement.split(per_chunk) if window_complement is not None
                             else [None] * len(window_mz.split(per_chunk)))
        for qc, kc, vc, mc, pc, cc in zip(
            q.split(per_chunk), k.split(per_chunk), v.split(per_chunk),
            window_mz.split(per_chunk), window_padding.split(per_chunk), complement_chunks,
        ):
            window_ids = torch.arange(offset, offset + len(qc), device=peaks.device)
            owners = window_ids // windows
            left = (window_ids % windows) * self.window - shift
            # A landmark already in this window must not receive a duplicate vote.
            duplicate = ((anchor_indices[owners] >= left[:, None]) &
                         (anchor_indices[owners] < left[:, None] + self.window))
            keys = torch.cat([kc, anchor_k[owners]], dim=2)
            values = torch.cat([vc, anchor_v[owners]], dim=2)
            key_mz = torch.cat([mc, anchor_mz[owners]], dim=1)
            key_padding = torch.cat([pc, anchor_padding[owners] | duplicate], dim=1)
            if neighbors is not None:
                # Share the union of query-selected partners within each window.
                # At most window * complement_neighbors extra keys; never N*N.
                indices = neighbors[offset:offset + len(qc)].sort(dim=1).values
                invalid = (indices < 0) | ((indices >= left[:, None]) &
                                           (indices < left[:, None] + self.window))
                invalid[:, 1:] |= indices[:, 1:] == indices[:, :-1]
                invalid |= ((indices[..., None] == anchor_indices[owners, None, :]) &
                            ~anchor_padding[owners, None, :]).any(-1)
                safe = indices.clamp_min(0)
                nk = full_k[owners[:, None], safe + shift].transpose(1, 2)
                nv = full_v[owners[:, None], safe + shift].transpose(1, 2)
                keys = torch.cat([keys, nk], dim=2)
                values = torch.cat([values, nv], dim=2)
                key_mz = torch.cat([key_mz, mz[owners[:, None], safe]], dim=1)
                key_padding = torch.cat([key_padding, invalid], dim=1)
            pieces.append(self._run_attention(qc, keys, values, mc, key_mz, key_padding, cc))
            offset += len(qc)
        value = torch.cat(pieces, dim=0).reshape(b, windows, self.heads, self.window, self.dim)
        value = value.permute(0, 1, 3, 2, 4).reshape(b, padded_length, d)
        return self.out(value[:, shift:shift + n])

    def forward_peaks(self, peaks, mz, padding, intensity, shift, complement=None):
        if (self.complement_bias is not None or self.complement_neighbors) and complement is None:
            raise ValueError("This model needs complement_mz inputs")
        if self.complement_bias is None and not self.complement_neighbors:
            complement = None
        if peaks.shape[1] <= self.full_limit:
            return self._full(peaks, mz, padding, complement)
        update = self._windows(peaks, mz, padding, intensity, shift, complement)
        if self.full_limit:
            lengths = (~padding).sum(1)
            short = torch.nonzero(lengths <= self.full_limit).flatten()
            if short.numel():
                length = int(lengths[short].max())
                full_update = self._full(peaks[short, :length], mz[short, :length], padding[short, :length],
                                         complement[short, :length] if complement is not None else None)
                update = update.clone()
                update[short, :length] = full_update
        return update


class XLBlock(nn.Module):
    def __init__(self, config, shift):
        super().__init__()
        self.window, self.shift = config.local_window, shift
        self.global_attn = Attention(config)
        self.peak_global_attn = Attention(config)
        self.local_attn = MassAwarePeakAttention(config)
        self.norm_peak = nn.LayerNorm(config.d_model)
        self.norm_global = nn.LayerNorm(config.d_model)
        self.norm_local = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.ff_multiplier), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.ff_multiplier, config.d_model),
        )

    def forward(self, peaks, global_tokens, padding, global_padding, mz, intensity, complement=None):
        p, g = self.norm_peak(peaks), self.norm_global(global_tokens)
        memory = torch.cat([g, p], 1)
        mask = torch.cat([global_padding, padding], 1)
        global_tokens = global_tokens + self.drop(self.global_attn(g, memory, mask))
        peaks = peaks + self.drop(self.peak_global_attn(p, self.norm_global(global_tokens), global_padding))
        local = self.local_attn.forward_peaks(self.norm_local(peaks), mz, padding, intensity, self.shift, complement)
        peaks = peaks + self.drop(local)
        peaks = peaks + self.drop(self.ff(self.norm_ff(peaks)))
        return peaks.masked_fill(padding[..., None], 0), global_tokens


class XLPeakClassifier(nn.Module):
    def __init__(self, config: XLConfig):
        super().__init__()
        if config.d_model < 1 or config.num_heads < 1 or config.d_model % config.num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if min(config.num_layers, config.local_window, config.global_slots, config.mz_frequencies) < 1:
            raise ValueError("Layer, window, slot and frequency counts must be positive")
        if config.full_attention_max_peaks < 0:
            raise ValueError("full_attention_max_peaks must be nonnegative")
        if min(config.pairwise_mass_frequencies, config.pairwise_mass_hidden_dim, config.pairwise_query_chunk_size) < 1:
            raise ValueError("Pairwise frequency, hidden and chunk sizes must be positive")
        if config.mass_attention_landmarks < 0:
            raise ValueError("mass_attention_landmarks must be nonnegative")
        if (not isinstance(config.complement_neighbors, int) or config.complement_neighbors < 0
                or not math.isfinite(config.complement_tolerance_da) or config.complement_tolerance_da <= 0):
            raise ValueError("complement_neighbors must be a nonnegative integer and tolerance must be positive")
        if not 0 <= config.dropout < 1 or not 0 < config.mz_min_wavelength < config.mz_max_wavelength:
            raise ValueError("Invalid dropout or m/z wavelengths")
        self.config = config
        self.mass_embedding = MassEmbedding(config)
        self.delta_embedding = MassEmbedding(config)
        self.peak_projection = nn.Linear(config.peak_input_dim, config.d_model)
        self.intensity_embedding = nn.Sequential(nn.Linear(2, config.d_model), nn.GELU(),
                                                nn.Linear(config.d_model, config.d_model))
        self.context_projection = nn.Sequential(nn.Linear(config.context_input_dim, config.d_model),
                                               nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.context_position = nn.Parameter(torch.randn(1, config.max_context_tokens, config.d_model) * 0.02)
        self.slots = nn.Parameter(torch.randn(1, config.global_slots, config.d_model) * 0.02)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([XLBlock(config, 0 if i % 2 == 0 else config.local_window // 2)
                                     for i in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(config.d_model, 1))

    def forward(self, peak_features, mz, delta_mz, intensity_features,
                context, context_mask, padding_mask, complement_mz=None):
        if context.shape[1] > self.config.max_context_tokens:
            raise ValueError("Too many context spectra for this checkpoint")
        peaks = self.input_norm(self.peak_projection(peak_features)
                                + self.mass_embedding(mz) + self.delta_embedding(delta_mz)
                                + self.intensity_embedding(intensity_features))
        peaks = peaks.masked_fill(padding_mask[..., None], 0)
        spectrum = self.context_projection(context) + self.context_position[:, :context.shape[1]]
        spectrum = spectrum.masked_fill(context_mask[..., None], 0)
        global_tokens = torch.cat([spectrum, self.slots.expand(len(peaks), -1, -1)], 1)
        global_padding = F.pad(context_mask, (0, self.config.global_slots), value=False)
        for block in self.blocks:
            if self.training and self.config.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                peaks, global_tokens = checkpoint(block, peaks, global_tokens, padding_mask,
                                                   global_padding, mz, intensity_features[..., 1], complement_mz,
                                                   use_reentrant=False)
            else:
                peaks, global_tokens = block(peaks, global_tokens, padding_mask, global_padding,
                                             mz, intensity_features[..., 1], complement_mz)
        summary = global_tokens[:, :1].expand(-1, peaks.shape[1], -1)
        logits = self.head(torch.cat([self.output_norm(peaks), summary], -1)).squeeze(-1)
        return logits.masked_fill(padding_mask, 0)


MODEL_INPUTS = ("peak_features", "mz", "delta_mz", "intensity_features", "context", "context_mask",
                "padding_mask", "complement_mz")


def model_inputs(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in MODEL_INPUTS if key in batch}


def amp_context(device, training_config):
    """Use the same compute-precision policy for validation and deployment."""
    if device.type != "cuda" or not training_config.get("enable_amp", True):
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)
