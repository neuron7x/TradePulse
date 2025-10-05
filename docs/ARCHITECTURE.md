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
