# Architecture Overview
Contracts-first via protobuf. Go engines (VPIN, orderbook, regime) + Python execution loop.
Next.js dashboard consumes gRPC-web gateway (to be added).

## Indicator/Feature Fractal Pattern

To make the "fractal" composition of indicators explicit, every transformer now
follows two canonical interfaces:

- `BaseFeature`: single, pure transformer that turns raw input into a
  `FeatureResult` (value + metadata). All quantitative indicators inherit from
  this base, so a new indicator only needs to implement `transform(...)`.
- `BaseBlock`: container that orchestrates homogeneous features. The concrete
  `FeatureBlock` simply loops over registered features and merges their outputs,
  but blocks can themselves be nested, giving the repeating structure the
  methodology requires.

Any future indicator/alpha block that respects these contracts can be dropped
into higher-level orchestrations (phase detector, policy router, dashboards)
without additional glue code. This rule is now part of the architecture
checklist alongside protobuf contracts and gRPC integration.

## Domain isolation

The trading primitives (signals, orders, positions) now live in the dedicated
`domain/` package. The package defines:

- `Signal` and `SignalAction` – immutable strategy outputs with confidence
  bounds and metadata validation.
- `Order`, `OrderSide`, `OrderType`, `OrderStatus` – an aggregate that enforces
  lifecycle constraints, fill handling, and type-safe enumerations.
- `Position` – exposure management with mark-to-market and realized PnL
  updates.

Domain code is strictly isolated from UI and infrastructure concerns. UI or
presentation layers must consume domain objects through DTO helpers located in
`application/`. Imports from UI into `domain/` are forbidden, and any new
business rules must be implemented inside the domain layer to keep testing
simple and dependencies minimal.
