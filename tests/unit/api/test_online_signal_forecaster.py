"""Unit tests for :class:`OnlineSignalForecaster`."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

# Ensure module import does not attempt to build the FastAPI app during tests.
os.environ.setdefault("TRADEPULSE_ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "test-audit-secret")

from application.api.service import OnlineSignalForecaster, PredictionResponse
from application.trading import signal_to_dto
from domain.signal import SignalAction


pytestmark = pytest.mark.filterwarnings(
    "ignore:'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.:DeprecationWarning"
)


class _StubPipeline:
    """Lightweight pipeline stub used to bypass the heavy feature pipeline."""

    def __init__(self, transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> None:
        self._transform_fn = transform_fn or (lambda frame: frame)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - trivial
        return self._transform_fn(frame)


@pytest.fixture
def make_forecaster() -> Callable[[pd.DataFrame], OnlineSignalForecaster]:
    def _factory(features: pd.DataFrame) -> OnlineSignalForecaster:
        pipeline = _StubPipeline(lambda _frame: features.copy())
        return OnlineSignalForecaster(pipeline=pipeline)

    return _factory


class TestLatestFeatureVector:
    def test_empty_frame_raises_bad_request(self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]) -> None:
        forecaster = make_forecaster(pd.DataFrame())

        with pytest.raises(HTTPException) as excinfo:
            forecaster.latest_feature_vector(pd.DataFrame())

        assert excinfo.value.status_code == 400

    def test_all_nan_row_raises_unprocessable_entity(
        self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        features = pd.DataFrame([{"macd": np.nan, "rsi": np.nan, "return_1": np.nan}])
        forecaster = make_forecaster(features)

        with pytest.raises(HTTPException) as excinfo:
            forecaster.latest_feature_vector(features)

        assert excinfo.value.status_code == 422

    def test_valid_row_returns_clean_series(
        self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        features = pd.DataFrame(
            [
                {
                    "macd": 0.1,
                    "macd_signal": 0.05,
                    "macd_histogram": 0.05,
                    "macd_ema_fast": 101.0,
                    "macd_ema_slow": 100.0,
                    "ema_5": 100.5,
                    "ema_20": 100.2,
                    "ema_60": 99.8,
                    "rsi": 53.0,
                    "return_1": 0.005,
                    "queue_imbalance": 0.2,
                    "volatility_20": 0.01,
                }
            ]
        )
        forecaster = make_forecaster(features)

        latest = forecaster.latest_feature_vector(features)

        assert isinstance(latest, pd.Series)
        assert not latest.isna().any()
        assert {"macd_signal", "macd_histogram", "macd_ema_fast", "macd_ema_slow"}.issubset(
            latest.index
        )
        assert {key for key in latest.index if key.startswith("ema_")} >= {"ema_5", "ema_20", "ema_60"}


class TestDeriveSignal:
    @pytest.mark.parametrize(
        "series, expected_action",
        [
            (
                pd.Series(
                    {
                        "macd": 1.2,
                        "macd_signal": 0.8,
                        "macd_histogram": 0.4,
                        "macd_ema_fast": 102.0,
                        "macd_ema_slow": 99.0,
                        "ema_5": 101.5,
                        "ema_20": 100.0,
                        "ema_60": 98.5,
                        "rsi": 68.0,
                        "return_1": 0.03,
                        "queue_imbalance": 0.8,
                        "volatility_20": 0.05,
                    }
                ),
                SignalAction.BUY,
            ),
            (
                pd.Series(
                    {
                        "macd": -1.5,
                        "macd_signal": -1.0,
                        "macd_histogram": -0.5,
                        "macd_ema_fast": 97.0,
                        "macd_ema_slow": 100.0,
                        "ema_5": 98.0,
                        "ema_20": 99.5,
                        "ema_60": 101.0,
                        "rsi": 28.0,
                        "return_1": -0.04,
                        "queue_imbalance": -0.7,
                        "volatility_20": 0.03,
                    }
                ),
                SignalAction.SELL,
            ),
            (
                pd.Series(
                    {
                        "macd": 0.02,
                        "macd_signal": 0.01,
                        "macd_histogram": 0.01,
                        "macd_ema_fast": 100.0,
                        "macd_ema_slow": 100.0,
                        "ema_5": 100.0,
                        "ema_20": 100.0,
                        "ema_60": 100.0,
                        "rsi": 49.0,
                        "return_1": 0.0005,
                        "queue_imbalance": 0.01,
                        "volatility_20": 0.02,
                    }
                ),
                SignalAction.HOLD,
            ),
        ],
    )
    def test_action_and_confidence_bounds(
        self, series: pd.Series, expected_action: SignalAction, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        forecaster = make_forecaster(pd.DataFrame([series]))
        horizon = 900

        signal, score = forecaster.derive_signal("BTC-USD", series, horizon)

        assert signal.action == expected_action
        assert signal.metadata["horizon_seconds"] == horizon
        assert pytest.approx(signal.metadata["score"], rel=1e-6) == score
        assert 0.0 <= signal.confidence <= 1.0

        if expected_action is SignalAction.BUY:
            assert score > 0.15
        elif expected_action is SignalAction.SELL:
            assert score < -0.15
        else:
            assert -0.15 <= score <= 0.15

    def test_signal_to_dto_in_prediction_response(
        self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        features = pd.DataFrame(
            [
                {
                    "macd": 0.5,
                    "macd_signal": 0.2,
                    "macd_histogram": 0.3,
                    "macd_ema_fast": 103.0,
                    "macd_ema_slow": 100.5,
                    "ema_5": 102.5,
                    "ema_20": 101.0,
                    "ema_60": 99.5,
                    "rsi": 62.0,
                    "return_1": 0.015,
                    "queue_imbalance": 0.4,
                    "volatility_20": 0.01,
                }
            ]
        )
        series = features.iloc[0]
        horizon = 600
        forecaster = make_forecaster(features)

        signal, score = forecaster.derive_signal("ETH-USD", series, horizon)
        response = PredictionResponse(
            symbol="ETH-USD",
            horizon_seconds=horizon,
            score=score,
            signal=signal_to_dto(signal),
        )

        assert response.horizon_seconds == horizon
        metadata = response.signal["metadata"]
        assert metadata["horizon_seconds"] == horizon
        assert metadata["score"] == pytest.approx(score, rel=1e-6)
        indicators = metadata["technical_indicators"]
        assert indicators["macd_signal"] == features.iloc[0]["macd_signal"]
        assert indicators["macd_histogram"] == features.iloc[0]["macd_histogram"]
        assert indicators["emas"]["ema_20"] == features.iloc[0]["ema_20"]

    def test_macd_divergence_biases_signal_buy(
        self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        series = pd.Series(
            {
                "macd": 1.0,
                "macd_signal": 0.1,
                "macd_histogram": 1.2,
                "macd_ema_fast": 105.0,
                "macd_ema_slow": 100.0,
                "ema_5": 104.0,
                "ema_20": 102.0,
                "ema_60": 99.0,
                "rsi": 50.0,
                "return_1": 0.0,
                "queue_imbalance": 0.0,
                "volatility_20": 0.01,
            }
        )
        forecaster = make_forecaster(pd.DataFrame([series]))

        signal, score = forecaster.derive_signal("SOL-USD", series, 300)

        assert signal.action is SignalAction.BUY
        assert score > 0.15

    def test_macd_divergence_biases_signal_sell(
        self, make_forecaster: Callable[[pd.DataFrame], OnlineSignalForecaster]
    ) -> None:
        series = pd.Series(
            {
                "macd": -1.0,
                "macd_signal": -0.1,
                "macd_histogram": -1.3,
                "macd_ema_fast": 95.0,
                "macd_ema_slow": 100.0,
                "ema_5": 96.0,
                "ema_20": 98.0,
                "ema_60": 101.0,
                "rsi": 50.0,
                "return_1": 0.0,
                "queue_imbalance": 0.0,
                "volatility_20": 0.01,
            }
        )
        forecaster = make_forecaster(pd.DataFrame([series]))

        signal, score = forecaster.derive_signal("SOL-USD", series, 300)

        assert signal.action is SignalAction.SELL
        assert score < -0.15

