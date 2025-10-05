# Indicators

- Kuramoto order parameter R
- Shannon entropy H and ΔH
- Hurst exponent
- Ricci curvature (graph-based)

## Fractal feature/block pattern

All indicators implement the `BaseFeature` interface and can be registered in a
`FeatureBlock`. Blocks can host features or other blocks, so indicator
pipelines (phase → regime → policy) repeat the same contract at every level.
This keeps the fractal structure explicit: adding a new indicator only requires
defining `transform(...)` and (optionally) plugging it into an existing block.
