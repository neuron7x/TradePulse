# tradepulse-accel

Rust acceleration crate for TradePulse numeric primitives.

This crate exposes sliding window extraction, quantile computation, and 1D convolution
helpers via [PyO3](https://pyo3.rs/) and is packaged with
[`maturin`](https://github.com/PyO3/maturin).

## Building for Python

The crate ships as the optional ``tradepulse_accel`` Python extension. To build it
locally (for example when working on the TradePulse monorepo) run:

```bash
cd rust/tradepulse-accel
maturin develop --release
```

Once installed the Python package automatically dispatches to the Rust implementation
whenever it is importable. If the extension is missing, the high level APIs fall back to
NumPy or pure-Python implementations so the platform remains fully functional.
