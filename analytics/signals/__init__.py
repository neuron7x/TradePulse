"""Signal research utilities for feature engineering and model evaluation."""

from .pipeline import (
    FeaturePipelineConfig,
    LeakageGate,
    ModelCandidate,
    SignalFeaturePipeline,
    SignalModelEvaluation,
    SignalModelSelector,
    build_supervised_learning_frame,
    make_default_candidates,
)

__all__ = [
    "FeaturePipelineConfig",
    "LeakageGate",
    "ModelCandidate",
    "SignalFeaturePipeline",
    "SignalModelEvaluation",
    "SignalModelSelector",
    "build_supervised_learning_frame",
    "make_default_candidates",
]
