"""Inventory management utilities for cross-venue capital orchestration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from threading import RLock
from time import perf_counter_ns
from typing import Dict, Mapping

from .liquidity import LiquidityLedger, LiquiditySnapshot
from .metrics import LatencyTracker
from .models import CapitalTransferPlan


class InventoryError(RuntimeError):
    """Raised when inventory state cannot be analysed or rebalanced."""


def _timedelta_to_milliseconds(value: timedelta) -> float:
    return value.total_seconds() * 1000.0


class InventoryLatencyStage(Enum):
    """Stages of the inventory workflow used for latency monitoring."""

    SNAPSHOT = "snapshot"
    TARGET_COMPUTATION = "target_computation"
    REBALANCE = "rebalance"
    ALLOCATION = "allocation"


@dataclass(slots=True, frozen=True)
class LatencyThreshold:
    """Latency limits applied to a given inventory stage."""

    p95: timedelta
    hard_limit: timedelta
    min_samples: int = 32

    def enforce(self, stage: InventoryLatencyStage, stats: "InventoryLatencyStats") -> None:
        if stats.sample_size == 0:
            return
        if stats.max_latency > self.hard_limit:
            raise InventoryError(
                "{stage} latency {observed:.3f} ms exceeded hard limit {limit:.3f} ms".format(
                    stage=stage.value,
                    observed=_timedelta_to_milliseconds(stats.max_latency),
                    limit=_timedelta_to_milliseconds(self.hard_limit),
                )
            )
        if stats.sample_size >= self.min_samples and stats.p95 > self.p95:
            raise InventoryError(
                "{stage} latency p95 {observed:.3f} ms breached budget {limit:.3f} ms".format(
                    stage=stage.value,
                    observed=_timedelta_to_milliseconds(stats.p95),
                    limit=_timedelta_to_milliseconds(self.p95),
                )
            )


@dataclass(slots=True, frozen=True)
class InventoryLatencyStats:
    """Latency measurements summarising a workflow stage."""

    last: timedelta
    p50: timedelta
    p95: timedelta
    p99: timedelta
    max_latency: timedelta
    sample_size: int

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "last_ms": _timedelta_to_milliseconds(self.last),
            "p50_ms": _timedelta_to_milliseconds(self.p50),
            "p95_ms": _timedelta_to_milliseconds(self.p95),
            "p99_ms": _timedelta_to_milliseconds(self.p99),
            "max_ms": _timedelta_to_milliseconds(self.max_latency),
            "samples": self.sample_size,
        }


@dataclass(slots=True, frozen=True)
class InventoryLatencyReport:
    """Materialised latency snapshot suitable for CI reporting."""

    generated_at: datetime
    stage_stats: Dict[InventoryLatencyStage, InventoryLatencyStats]

    def as_dict(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "stages": {
                stage.value: stats.as_dict() for stage, stats in self.stage_stats.items()
            },
        }


@dataclass(slots=True, frozen=True)
class InventoryLatencyBudget:
    """Latency budgets for key stages of the inventory workflow."""

    snapshot: LatencyThreshold
    target_computation: LatencyThreshold
    rebalance: LatencyThreshold
    allocation: LatencyThreshold

    @classmethod
    def default(cls) -> "InventoryLatencyBudget":
        return cls(
            snapshot=LatencyThreshold(
                p95=timedelta(milliseconds=5),
                hard_limit=timedelta(milliseconds=20),
            ),
            target_computation=LatencyThreshold(
                p95=timedelta(milliseconds=3),
                hard_limit=timedelta(milliseconds=15),
            ),
            rebalance=LatencyThreshold(
                p95=timedelta(milliseconds=6),
                hard_limit=timedelta(milliseconds=25),
            ),
            allocation=LatencyThreshold(
                p95=timedelta(milliseconds=4),
                hard_limit=timedelta(milliseconds=18),
            ),
        )

    def threshold_for(self, stage: InventoryLatencyStage) -> LatencyThreshold:
        if stage is InventoryLatencyStage.SNAPSHOT:
            return self.snapshot
        if stage is InventoryLatencyStage.TARGET_COMPUTATION:
            return self.target_computation
        if stage is InventoryLatencyStage.REBALANCE:
            return self.rebalance
        if stage is InventoryLatencyStage.ALLOCATION:
            return self.allocation
        raise ValueError(f"Unknown inventory latency stage {stage!r}")


class InventoryLatencyMonitor:
    """Thread-safe latency tracker for inventory workflows."""

    def __init__(self, *, window: int = 512) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._trackers: Dict[InventoryLatencyStage, LatencyTracker] = {
            stage: LatencyTracker(max_samples=window) for stage in InventoryLatencyStage
        }
        self._last: Dict[InventoryLatencyStage, timedelta] = {}
        self._lock = RLock()

    @contextmanager
    def track(self, stage: InventoryLatencyStage):
        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start
            duration = timedelta(seconds=elapsed_ns / 1_000_000_000)
            self.record_sample(stage, duration)

    def record_sample(self, stage: InventoryLatencyStage, duration: timedelta) -> None:
        if duration < timedelta(0):
            raise ValueError("duration must be non-negative")
        with self._lock:
            tracker = self._trackers[stage]
            tracker.record(duration)
            self._last[stage] = duration

    def latest_stats(self, stage: InventoryLatencyStage) -> InventoryLatencyStats:
        with self._lock:
            tracker = self._trackers[stage]
            last = self._last.get(stage, timedelta(0))
            if len(tracker) == 0:
                return InventoryLatencyStats(
                    last=last,
                    p50=timedelta(0),
                    p95=timedelta(0),
                    p99=timedelta(0),
                    max_latency=timedelta(0),
                    sample_size=0,
                )
            return InventoryLatencyStats(
                last=last,
                p50=tracker.percentile(50),
                p95=tracker.percentile(95),
                p99=tracker.percentile(99),
                max_latency=tracker.max_latency(),
                sample_size=len(tracker),
            )

    def report(self) -> InventoryLatencyReport:
        stats = {
            stage: self.latest_stats(stage) for stage in InventoryLatencyStage
        }
        return InventoryLatencyReport(
            generated_at=datetime.now(timezone.utc),
            stage_stats=stats,
        )


def export_inventory_latency_report(
    report: InventoryLatencyReport,
    *,
    directory: Path | str = Path("reports/performance"),
    filename: str | None = None,
) -> Path:
    """Persist a latency report for CI artefact collection."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = (
            "inventory_latency_"
            + report.generated_at.strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
    path = target_dir / filename
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return path


def _to_decimal(value: Decimal | int | str | float) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    raise TypeError(f"Unsupported numeric type {type(value)!r}")


@dataclass(slots=True, frozen=True)
class InventoryTarget:
    """Target weighting and buffers for a venue's inventory."""

    target_weight: Decimal
    min_base_buffer: Decimal = Decimal("0")
    min_quote_buffer: Decimal = Decimal("0")
    max_weight: Decimal | None = None

    def __post_init__(self) -> None:
        share = _to_decimal(self.target_weight)
        if share < Decimal("0"):
            raise ValueError("target_weight must be non-negative")
        object.__setattr__(self, "target_weight", share)
        base_buffer = _to_decimal(self.min_base_buffer)
        if base_buffer < Decimal("0"):
            raise ValueError("min_base_buffer must be non-negative")
        object.__setattr__(self, "min_base_buffer", base_buffer)
        quote_buffer = _to_decimal(self.min_quote_buffer)
        if quote_buffer < Decimal("0"):
            raise ValueError("min_quote_buffer must be non-negative")
        object.__setattr__(self, "min_quote_buffer", quote_buffer)
        if self.max_weight is not None:
            max_share = _to_decimal(self.max_weight)
            if max_share <= Decimal("0"):
                raise ValueError("max_weight must be positive when provided")
            object.__setattr__(self, "max_weight", max_share)


@dataclass(slots=True, frozen=True)
class VenueInventory:
    """Computed inventory statistics for a single venue."""

    exchange_id: str
    snapshot: LiquiditySnapshot
    target_weight: Decimal
    desired_base: Decimal
    surplus: Decimal
    deficit: Decimal


@dataclass(slots=True, frozen=True)
class InventorySnapshot:
    """Aggregated view of a symbol's inventory across venues."""

    symbol: str
    base_asset: str
    quote_asset: str
    total_base: Decimal
    total_quote: Decimal
    venues: tuple[VenueInventory, ...]

    def is_balanced(self, tolerance: Decimal, min_transfer: Decimal) -> bool:
        threshold = max(tolerance * self.total_base, min_transfer)
        for venue in self.venues:
            if venue.surplus > threshold or venue.deficit > threshold:
                return False
        return True


@dataclass(slots=True, frozen=True)
class RebalanceLeg:
    """Single transfer leg within a rebalance plan."""

    source_exchange: str
    target_exchange: str
    asset: str
    amount: Decimal
    unit_cost: Decimal


@dataclass(slots=True, frozen=True)
class RebalancePlan:
    """Recommended sequence of transfers to restore balance."""

    symbol: str
    asset: str
    transfers: tuple[RebalanceLeg, ...]
    estimated_cost: Decimal

    def to_transfer_plan(
        self,
        transfer_id: str,
        *,
        initiated_at: datetime | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> CapitalTransferPlan:
        if not self.transfers:
            raise InventoryError("Cannot materialise a transfer plan without legs")
        if initiated_at is None:
            initiated_at = datetime.now(timezone.utc)
        legs: Dict[tuple[str, str], Decimal] = {}
        for leg in self.transfers:
            legs[(leg.source_exchange, self.asset)] = (
                legs.get((leg.source_exchange, self.asset), Decimal("0")) + leg.amount
            )
            legs[(leg.target_exchange, self.asset)] = (
                legs.get((leg.target_exchange, self.asset), Decimal("0")) + leg.amount
            )
        meta: Dict[str, str] = {
            "symbol": self.symbol,
            "asset": self.asset,
            "estimated_cost": str(self.estimated_cost),
        }
        if metadata:
            meta.update(metadata)
        return CapitalTransferPlan(
            transfer_id=transfer_id,
            legs=legs,
            initiated_at=initiated_at,
            metadata=meta,
        )


class InventoryManager:
    """Orchestrates inventory monitoring and rebalance suggestions."""

    def __init__(
        self,
        ledger: LiquidityLedger,
        pair_config: Mapping[str, tuple[str, str]],
        *,
        rebalance_tolerance: Decimal = Decimal("0.02"),
        min_transfer: Decimal = Decimal("0"),
        transfer_costs: Mapping[tuple[str, str], Decimal] | None = None,
        latency_monitor: InventoryLatencyMonitor | None = None,
        latency_budget: InventoryLatencyBudget | None = None,
    ) -> None:
        if rebalance_tolerance < Decimal("0"):
            raise ValueError("rebalance_tolerance must be non-negative")
        if min_transfer < Decimal("0"):
            raise ValueError("min_transfer must be non-negative")
        self._ledger = ledger
        self._pair_config = dict(pair_config)
        self._tolerance = rebalance_tolerance
        self._min_transfer = min_transfer
        self._transfer_costs: Dict[tuple[str, str], Decimal] = {}
        if transfer_costs:
            for key, cost in transfer_costs.items():
                self._transfer_costs[key] = _to_decimal(cost)
        self._latency_monitor = latency_monitor or InventoryLatencyMonitor()
        self._latency_budget = latency_budget or InventoryLatencyBudget.default()

    def snapshot(
        self,
        symbol: str,
        targets: Mapping[str, InventoryTarget],
    ) -> InventorySnapshot:
        if symbol not in self._pair_config:
            raise InventoryError(f"Unknown symbol {symbol}")
        base_asset, quote_asset = self._pair_config[symbol]
        if not targets:
            raise InventoryError("At least one target must be specified")
        weights = self._normalise_weights(targets)
        with self._latency_monitor.track(InventoryLatencyStage.SNAPSHOT):
            venues: list[VenueInventory] = []
            total_base = Decimal("0")
            total_quote = Decimal("0")
            for exchange_id, target in targets.items():
                snapshot = self._ledger.get_snapshot(exchange_id, symbol)
                if snapshot is None:
                    raise InventoryError(
                        f"No liquidity snapshot available for {exchange_id}:{symbol}"
                    )
                weight = weights[exchange_id]
                base_available = snapshot.base_available
                quote_available = snapshot.quote_available
                total_base += base_available
                total_quote += quote_available
                venue = VenueInventory(
                    exchange_id=exchange_id,
                    snapshot=snapshot,
                    target_weight=weight,
                    desired_base=Decimal("0"),
                    surplus=Decimal("0"),
                    deficit=Decimal("0"),
                )
                venues.append(venue)

            venues_with_targets = self._compute_targets(
                venues, targets, weights, total_base
            )
            snapshot_view = InventorySnapshot(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                total_base=total_base,
                total_quote=total_quote,
                venues=tuple(venues_with_targets),
            )
        self._enforce_latency(InventoryLatencyStage.SNAPSHOT)
        return snapshot_view

    def propose_rebalance(
        self,
        symbol: str,
        targets: Mapping[str, InventoryTarget],
    ) -> tuple[InventorySnapshot, RebalancePlan | None]:
        with self._latency_monitor.track(InventoryLatencyStage.REBALANCE):
            snapshot = self.snapshot(symbol, targets)
            plan: RebalancePlan | None = None
            if snapshot.total_base > Decimal("0"):
                threshold = max(
                    self._tolerance * snapshot.total_base, self._min_transfer
                )
                if not snapshot.is_balanced(self._tolerance, self._min_transfer):
                    surplus_map: Dict[str, Decimal] = {}
                    deficit_map: Dict[str, Decimal] = {}
                    for venue in snapshot.venues:
                        if venue.surplus > threshold:
                            surplus_map[venue.exchange_id] = venue.surplus
                        if venue.deficit > threshold:
                            deficit_map[venue.exchange_id] = venue.deficit
                    if surplus_map and deficit_map:
                        transfers = self._allocate_transfers(
                            snapshot.base_asset,
                            surplus_map,
                            deficit_map,
                            threshold,
                        )
                        if transfers:
                            estimated_cost = sum(
                                (
                                    leg.amount * leg.unit_cost
                                    for leg in transfers
                                ),
                                Decimal("0"),
                            )
                            plan = RebalancePlan(
                                symbol=symbol,
                                asset=snapshot.base_asset,
                                transfers=tuple(transfers),
                                estimated_cost=estimated_cost,
                            )
        self._enforce_latency(InventoryLatencyStage.REBALANCE)
        return snapshot, plan

    def _normalise_weights(
        self, targets: Mapping[str, InventoryTarget]
    ) -> Dict[str, Decimal]:
        weights: Dict[str, Decimal] = {}
        total = Decimal("0")
        for exchange_id, target in targets.items():
            weight = target.target_weight
            total += weight
            weights[exchange_id] = weight
        if total <= Decimal("0"):
            raise InventoryError("Target weights must sum to a positive value")
        for exchange_id, value in list(weights.items()):
            weights[exchange_id] = value / total
        return weights

    def _compute_targets(
        self,
        venues: list[VenueInventory],
        targets: Mapping[str, InventoryTarget],
        weights: Mapping[str, Decimal],
        total_base: Decimal,
    ) -> list[VenueInventory]:
        with self._latency_monitor.track(InventoryLatencyStage.TARGET_COMPUTATION):
            if total_base <= Decimal("0"):
                result = venues
            else:
                updated: list[VenueInventory] = []
                for venue in venues:
                    target = targets[venue.exchange_id]
                    normalised_weight = weights[venue.exchange_id]
                    desired = total_base * normalised_weight
                    min_base = target.min_base_buffer
                    max_cap = target.max_weight
                    if max_cap is not None:
                        cap_amount = total_base * max_cap
                        if desired > cap_amount:
                            desired = cap_amount
                    available = venue.snapshot.base_available
                    surplus = Decimal("0")
                    deficit = Decimal("0")
                    if available > desired:
                        transferable = max(available - desired, Decimal("0"))
                        spare_after_buffer = max(available - min_base, Decimal("0"))
                        surplus = min(transferable, spare_after_buffer)
                        if venue.snapshot.quote_available < target.min_quote_buffer:
                            surplus = Decimal("0")
                    else:
                        target_amount = max(desired, min_base)
                        if available < target_amount:
                            deficit = target_amount - available
                    updated.append(
                        VenueInventory(
                            exchange_id=venue.exchange_id,
                            snapshot=venue.snapshot,
                            target_weight=normalised_weight,
                            desired_base=desired,
                            surplus=surplus,
                            deficit=deficit,
                        )
                    )
                result = updated
        self._enforce_latency(InventoryLatencyStage.TARGET_COMPUTATION)
        return result

    def _allocate_transfers(
        self,
        asset: str,
        surplus_map: Dict[str, Decimal],
        deficit_map: Dict[str, Decimal],
        threshold: Decimal,
    ) -> list[RebalanceLeg]:
        with self._latency_monitor.track(InventoryLatencyStage.ALLOCATION):
            transfers: list[RebalanceLeg] = []
            while surplus_map and deficit_map:
                best_pair = None
                best_cost = None
                for source, surplus in surplus_map.items():
                    for target, deficit in deficit_map.items():
                        if source == target:
                            continue
                        amount = min(surplus, deficit)
                        if amount <= threshold:
                            continue
                        cost = self._transfer_costs.get((source, target), Decimal("0"))
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_pair = (source, target, amount)
                if best_pair is None:
                    break
                source, target, amount = best_pair
                transfers.append(
                    RebalanceLeg(
                        source_exchange=source,
                        target_exchange=target,
                        asset=asset,
                        amount=amount,
                        unit_cost=best_cost or Decimal("0"),
                    )
                )
                surplus_map[source] -= amount
                deficit_map[target] -= amount
                if surplus_map[source] <= threshold:
                    surplus_map.pop(source)
                if deficit_map[target] <= threshold:
                    deficit_map.pop(target)
        self._enforce_latency(InventoryLatencyStage.ALLOCATION)
        return transfers

    def latency_report(self) -> InventoryLatencyReport:
        """Return the most recent latency statistics."""

        return self._latency_monitor.report()

    def export_latency_report(
        self,
        *,
        directory: Path | str = Path("reports/performance"),
        filename: str | None = None,
    ) -> Path:
        """Export the current latency metrics for CI consumption."""

        report = self.latency_report()
        return export_inventory_latency_report(
            report,
            directory=directory,
            filename=filename,
        )

    def _enforce_latency(self, stage: InventoryLatencyStage) -> None:
        threshold = self._latency_budget.threshold_for(stage)
        stats = self._latency_monitor.latest_stats(stage)
        threshold.enforce(stage, stats)

