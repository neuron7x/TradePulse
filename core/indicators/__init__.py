# SPDX-License-Identifier: MIT
"""Indicator package exports."""

from .base import BaseBlock, BaseFeature, FeatureBlock, FeatureResult, FunctionalFeature
from .entropy import DeltaEntropyFeature, EntropyFeature, delta_entropy, entropy
from .hurst import HurstFeature, hurst_exponent
from .kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    compute_phase_gpu,
    kuramoto_order,
    multi_asset_kuramoto,
)
from .multiscale_kuramoto import (
    MultiScaleKuramoto,
    MultiScaleKuramotoFeature,
    MultiScaleResult,
    TimeFrame,
    WaveletWindowSelector,
)
from .ricci import MeanRicciFeature, build_price_graph, local_distribution, mean_ricci, ricci_curvature_edge
from .temporal_ricci import (
    GraphSnapshot,
    PriceLevelGraphBuilder,
    TemporalRicciAnalyzer,
    TemporalRicciResult,
)
from .kuramoto_ricci_composite import (
    CompositeSignal,
    KuramotoRicciComposite,
    MarketPhase,
    TradePulseCompositeEngine,
)

LEGACY_EXPORTS = [
    "BaseBlock",
    "BaseFeature",
    "FeatureBlock",
    "FeatureResult",
    "FunctionalFeature",
    "entropy",
    "delta_entropy",
    "EntropyFeature",
    "DeltaEntropyFeature",
    "hurst_exponent",
    "HurstFeature",
    "compute_phase",
    "compute_phase_gpu",
    "kuramoto_order",
    "multi_asset_kuramoto",
    "KuramotoOrderFeature",
    "MultiAssetKuramotoFeature",
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
    "build_price_graph",
    "local_distribution",
    "ricci_curvature_edge",
    "mean_ricci",
    "MeanRicciFeature",
    "GraphSnapshot",
    "PriceLevelGraphBuilder",
    "TemporalRicciAnalyzer",
    "TemporalRicciResult",
]

ADDITIONAL_EXPORTS = [
    "CompositeSignal",
    "KuramotoRicciComposite",
    "MarketPhase",
    "TradePulseCompositeEngine",
]

__all__ = list(dict.fromkeys(LEGACY_EXPORTS + ADDITIONAL_EXPORTS))
