from .base import BaseBlock, BaseFeature, FeatureBlock
from .kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
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
    "KuramotoOrderFeature",
    "MultiAssetKuramotoFeature",
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