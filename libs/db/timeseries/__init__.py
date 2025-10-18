"""Time-series data management primitives for TradePulse."""

from .config import (
    AggregationSpec,
    BenchmarkWorkload,
    DimensionColumn,
    IngestionConnectorConfig,
    MeasureColumn,
    RetentionPolicy,
    RollupAggregation,
    RollupMaterialization,
    SLAMetric,
    TimeSeriesSchema,
)
from .clickhouse import (
    ClickHouseBackupPlanner,
    ClickHouseIndex,
    ClickHouseIngestionConnector,
    ClickHouseQueryBuilder,
    ClickHouseSchemaManager,
    ClickHouseSLAManager,
)
from .timescale import (
    TimescaleBackupPlanner,
    TimescaleIngestionConnector,
    TimescaleQueryBuilder,
    TimescaleSchemaManager,
    TimescaleSLAManager,
)
from .benchmarks import BenchmarkRunner

__all__ = [
    "AggregationSpec",
    "BenchmarkRunner",
    "BenchmarkWorkload",
    "ClickHouseBackupPlanner",
    "ClickHouseIndex",
    "ClickHouseIngestionConnector",
    "ClickHouseQueryBuilder",
    "ClickHouseSchemaManager",
    "ClickHouseSLAManager",
    "DimensionColumn",
    "IngestionConnectorConfig",
    "MeasureColumn",
    "RetentionPolicy",
    "RollupAggregation",
    "RollupMaterialization",
    "SLAMetric",
    "TimeSeriesSchema",
    "TimescaleBackupPlanner",
    "TimescaleIngestionConnector",
    "TimescaleQueryBuilder",
    "TimescaleSchemaManager",
    "TimescaleSLAManager",
]
