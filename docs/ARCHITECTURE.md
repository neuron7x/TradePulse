# Architecture Overview
Contracts-first via protobuf. Go engines (VPIN, orderbook, regime) + Python execution loop.
Next.js dashboard consumes gRPC-web gateway (to be added).

## Feature and Block Fractal Pattern

To make the indicator stack fractal and self-similar, every transformer is expressed in terms of two shared interfaces:

1. **`BaseFeature`** – the atomic transformation contract with `transform`, `metadata()` and vector coercion helpers. Any numerical signal (NumPy, pandas, iterator) can be converted into a standard 1-D array and fed through a feature. Features expose their configuration via immutable metadata so orchestration layers can introspect and chain them.
2. **`BaseBlock`** – a compositional unit that wires multiple features into a reusable pipeline. Blocks expose their own metadata tree so higher-level agents (backtests, live evaluators) can recursively inspect and reuse sub-structures.

### Architectural Rule

> **All new indicators, signal transforms, or composite pipelines _must_ inherit from `BaseFeature` or `BaseBlock`.** Functional helpers can exist for ergonomic calls, but the canonical implementation lives on the feature/block subclass.

Following this rule keeps the codebase fractal: a block can contain other blocks or features while exposing the same metadata surface. Downstream components (strategy builders, dashboards, research notebooks) can treat any transformer uniformly, which simplifies dynamic composition, dependency injection, and automated documentation.

### Practical Benefits

- **Pluggability:** new indicators auto-register by subclassing `BaseFeature` or `BaseBlock`; no custom glue is necessary.
- **Introspection:** consistent metadata trees allow visualisation or exporting of transformation graphs.
- **Testing Discipline:** feature-level tests assert atomic behaviour, while block tests validate orchestrated flows with the same interface.
