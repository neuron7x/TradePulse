import numpy as np
import pandas as pd
import pytest

from backtest.time_splits import PurgedKFoldTimeSeriesSplit, WalkForwardSplitter


def _sample_frame():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "label_end": dates + pd.Timedelta(days=15),
            "value": np.arange(len(dates)),
        }
    )
    return frame


def test_walk_forward_split_respects_windows():
    frame = _sample_frame()
    splitter = WalkForwardSplitter(
        train_window="180D",
        test_window="60D",
        step="60D",
        time_col="timestamp",
        label_end_col="label_end",
        embargo_pct=0.1,
    )
    splits = list(splitter.split(frame))
    assert splits, "Expected at least one split"
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        train_times = frame.loc[train_idx, "timestamp"]
        test_times = frame.loc[test_idx, "timestamp"]
        assert train_times.max() < test_times.min()
        # embargo ensures a buffer
        max_train_pos = frame.index.get_indexer_for(train_idx).max()
        min_test_pos = frame.index.get_indexer_for(test_idx).min()
        assert min_test_pos - max_train_pos >= 1


def test_purged_walk_forward_removes_overlaps():
    frame = _sample_frame()
    # Introduce overlap: extend the label end of an early observation into the future
    frame.loc[2, "label_end"] = frame.loc[5, "timestamp"]
    splitter = WalkForwardSplitter(
        train_window="180D",
        test_window="60D",
        step="60D",
        time_col="timestamp",
        label_end_col="label_end",
        embargo_pct=0.0,
    )
    for train_idx, test_idx in splitter.split(frame):
        test_end = frame.loc[test_idx, "label_end"].max()
        train_overlaps = frame.loc[train_idx, "label_end"] >= frame.loc[test_idx, "timestamp"].min()
        assert not train_overlaps.any()
        assert frame.loc[train_idx, "timestamp"].max() < test_end


def test_purged_kfold_applies_embargo():
    frame = _sample_frame()
    splitter = PurgedKFoldTimeSeriesSplit(
        n_splits=4,
        time_col="timestamp",
        label_end_col="label_end",
        embargo_pct=0.2,
    )
    frame = frame.reset_index(drop=True)
    n = len(frame)
    embargo_count = int(np.ceil(n * 0.2))
    for train_idx, test_idx in splitter.split(frame):
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        last_test = max(test_idx)
        embargo_range = range(last_test + 1, min(last_test + 1 + embargo_count, n))
        assert not any(i in train_idx for i in embargo_range)


@pytest.mark.parametrize("n_splits", [1, 0])
def test_purged_kfold_requires_at_least_two_splits(n_splits):
    with pytest.raises(ValueError):
        PurgedKFoldTimeSeriesSplit(n_splits=n_splits)
