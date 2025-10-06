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
    "build_price_graph",
    "local_distribution",
    "ricci_curvature_edge",
    "mean_ricci",
    "MeanRicciFeature",
]

ADVANCED_EXPORTS = [
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
    "GraphSnapshot",
    "PriceLevelGraphBuilder",
    "TemporalRicciAnalyzer",
    "TemporalRicciResult",
    "CompositeSignal",
    "KuramotoRicciComposite",
    "MarketPhase",
    "TradePulseCompositeEngine",
]

__all__ = LEGACY_EXPORTS + [
    name for name in ADVANCED_EXPORTS if name not in LEGACY_EXPORTS
]
