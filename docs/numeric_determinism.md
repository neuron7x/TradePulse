# Numeric Determinism Testing Playbook

Numeric determinism protects TradePulse from silent drifts in model outputs,
portfolio risk metrics, and trade execution forecasts. Small discrepancies can
snowball into different orders being sent to venues or invalidating
backtesting studies. This playbook outlines how we test for deterministic
behaviour across hardware, operating systems, compilers, and runtime
configurations, and how to react when we detect inconsistencies.

## Guiding Principles

1. **Test across the real deployment matrix.** Re-run the same workload on
   every supported CPU architecture, operating system, and container image to
   catch runtime-specific floating-point behaviour.
2. **Pin sources of entropy.** Seed RNGs, fix thread counts, and normalise
   environment variables so reproducibility starts before the first line of
   numerical code executes.
3. **Fail on divergence beyond justified tolerances.** Emit diagnostics that
   quantify how far two runs drift and the inputs that triggered the mismatch,
   rather than silently clamping to a tolerance.
4. **Continuously guard regressions.** Integrate determinism checks into CI so
   compiler upgrades, dependency bumps, or configuration changes cannot land
   without being validated.

## Environment and Runtime Controls

| Control | Why it matters | Implementation |
| --- | --- | --- |
| Thread pinning | BLAS/OpenMP libraries may schedule work differently per run when multiple worker threads are available, producing non-deterministic reductions. | Use `core.utils.determinism.apply_thread_determinism` at process start; the unit test `tests/unit/test_runtime_threads.py` asserts we export the pinned values. |
| RNG seeding | NumPy, Python, and domain-specific generators (e.g. Poisson processes) must emit the same stream on every platform. | Initialise seeds in `analytics.runner.set_random_seeds` before constructing models; supply explicit seeds to stochastic algorithms. |
| Environment snapshots | Capturing input arguments, dataset revisions, and code digests allows us to re-run identical workloads. | `analytics.environment_parity.compute_parameters_digest` and `StrategyRunSnapshot` normalise inputs and source digests for reproducibility audits. |

> **Tip:** Always capture the runtime metadata (Git SHA, compiler version,
> BLAS vendor) in produced reports so divergences can be explained quickly.

## Test Strategy

1. **Canonical Dataset Replays**
   - Maintain a small, immutable dataset per strategy family.
   - Compute canonical outputs (signals, PnL, attribution) and store them under
     version control as golden files.
   - During CI, replay the dataset and compare the outputs using strict
     absolute/relative tolerances (≤ 1e-12 for double precision analytics).
2. **Cross-Platform Execution Matrix**
   - Schedule nightly jobs that run the same suite on Linux x86, Linux ARM, and
     macOS runners.
   - Diff the resulting artifacts. Any difference above tolerance produces a
     blocking failure.
3. **Compiler and Optimization Coverage**
   - Build C/C++/Rust extensions with all supported compilers (e.g. GCC, Clang)
     and flags (`-O2`, `-O3`, vectorisation toggles).
   - Run smoke tests for each build to surface optimisation-induced drifts.
4. **Floating-Point Stress Scenarios**
   - Execute targeted fuzzers that focus on numerically unstable regions (very
     small/large magnitudes, ill-conditioned matrices, denormals).
   - Assert stability by monitoring condition numbers and ensuring pivoting or
     regularisation kicks in before catastrophic cancellation happens.

## Diagnostics for Divergences

When a mismatch is detected, collect the following artefacts automatically:

- **Input provenance:** dataset identifiers, feature window metadata, and
  parameter hashes.
- **Environment fingerprint:** kernel version, CPU flags, BLAS vendor, Python
  version, and compiler build strings.
- **Numeric deltas:** maximum absolute and relative error, alongside indices or
  timestamps where the divergence occurred.
- **Intermediate checkpoints:** optional snapshots of matrix factorizations or
  Monte Carlo paths help isolate the stage where results diverged.

These artefacts belong in CI job summaries and should be attached to incident
reports when escalated.

## Minimising Floating-Point Error

- Prefer **double precision (`float64`)** for analytics unless latency budgets
  dictate otherwise.
- Use **compensated summation** (Kahan, Neumaier) when aggregating large
  vectors of heterogeneous magnitudes.
- Apply **explicit ordering** for reductions (sort inputs, tree reductions) so
  hardware parallelism does not re-order additions unpredictably.
- Enable **denormal flushing** only when profiling proves it avoids performance
  cliffs without harming accuracy.
- Keep **algorithmic conditioning** in mind: add regularisers, clamp covariance
  matrices to maintain positive definiteness, and avoid subtracting nearly
  equal numbers.

## Governance and Automation

- **Pre-merge gates:** Add determinism checks to `.github/workflows/heavy-math.yml`
  so any drift blocks the pull request.
- **Release sign-off:** The production readiness checklist must include a
  deterministic replay of the latest golden dataset before tagging a release.
- **Observability hooks:** Emit Prometheus metrics for determinism failures and
  route them to the reliability on-call rotation.

## Escalation Protocol

1. Freeze deployments touching the affected models or execution components.
2. Bisect recent changes (dependencies, compilers, OS images) using the captured
   metadata.
3. If the issue roots in third-party libraries, raise upstream tickets and pin
   versions until a fix ships.
4. Document the incident postmortem, including new regression tests or tighter
   tolerances introduced to prevent recurrence.

By institutionalising these practices, TradePulse maintains trustworthy numeric
pipelines regardless of where computations run.
