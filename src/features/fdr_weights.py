from __future__ import annotations

from dataclasses import dataclass
from src.types import FDRWeightResult


@dataclass(slots=True)
class FDRWeightConfig:
    enabled: bool = True
    clip_min: float = 0.0
    clip_max: float = 0.01
    weight_min: float = 0.2
    mode: str = "linear"


class FDRWeightComputer:
    def __init__(self, config: FDRWeightConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.clip_max < self.config.clip_min:
            raise ValueError("clip_max must be greater than or equal to clip_min")
        if not 0.0 <= self.config.weight_min <= 1.0:
            raise ValueError("weight_min must be between 0.0 and 1.0")
        if self.config.mode not in {"linear"}:
            raise ValueError(f"unsupported FDR weighting mode: {self.config.mode}")

    def compute(self, fdr: float | None) -> FDRWeightResult:
        if not self.config.enabled:
            return FDRWeightResult(
                raw_fdr=fdr,
                clipped_fdr=fdr,
                weight=1.0,
            )
        if fdr is None:
            return FDRWeightResult(
                raw_fdr=None,
                clipped_fdr=None,
                weight=1.0,
            )

        raw_fdr = float(fdr)
        clipped_fdr = min(max(raw_fdr, self.config.clip_min), self.config.clip_max)

        if self.config.mode == "linear":
            weight = self._linear_weight(clipped_fdr)
        else: # we use for now only the linear, other optinons like exp can be used!
            raise ValueError(f"unsupported FDR weighting mode: {self.config.mode}")

        return FDRWeightResult(
            raw_fdr=raw_fdr,
            clipped_fdr=clipped_fdr,
            weight=weight,
        )

    def _linear_weight(self, clipped_fdr: float) -> float:
        clip_range = self.config.clip_max - self.config.clip_min
        if clip_range == 0.0:
            return 1.0

        normalized = (clipped_fdr - self.config.clip_min) / clip_range
        weight = 1.0 - normalized

        if weight < self.config.weight_min:
            weight = self.config.weight_min

        return float(weight)