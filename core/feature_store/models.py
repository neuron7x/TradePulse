"""Data models for feature store materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(slots=True)
class FeatureSet:
    """Representation of a feature dataframe with metadata."""

    name: str
    dataframe: pd.DataFrame
    entity_columns: Sequence[str]
    timestamp_column: str

    def __post_init__(self) -> None:
        missing = [col for col in self.entity_columns if col not in self.dataframe.columns]
        if missing:
            raise ValueError(f"Entity columns {missing} are missing from dataframe")
        if self.timestamp_column not in self.dataframe.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' missing from dataframe")
        self.dataframe = self.dataframe.copy()
        self.dataframe[self.timestamp_column] = pd.to_datetime(
            self.dataframe[self.timestamp_column], utc=True, errors="coerce"
        )
        if self.dataframe[self.timestamp_column].isna().any():
            raise ValueError("Timestamp column contains null values after conversion to datetime")

    @property
    def columns(self) -> List[str]:
        return list(self.dataframe.columns)

    def entity_keys(self) -> pd.Series:
        """Return the deterministic entity key for each row."""

        return self.dataframe[self.entity_columns].astype(str).agg("|".join, axis=1)

    def to_records(self) -> Iterable[dict]:
        """Yield dictionaries ready for online materialization."""

        return self.dataframe.to_dict(orient="records")

    def empty_like(self) -> "FeatureSet":
        """Return an empty feature set preserving the schema."""

        empty_df = self.dataframe.iloc[0:0].copy()
        return FeatureSet(
            name=self.name,
            dataframe=empty_df,
            entity_columns=list(self.entity_columns),
            timestamp_column=self.timestamp_column,
        )


__all__ = ["FeatureSet"]
