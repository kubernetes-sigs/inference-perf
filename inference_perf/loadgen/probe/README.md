# Capacity Probe

**Sweep rate selection is a measurement problem, and this package treats it
as one.** Instead of guessing a max rate from a transient burst, the probe
holds concurrency fixed, climbs a geometric ladder, and reports capacity
constants with confidence intervals. Everything in this directory is pure
measurement and estimation logic: no dispatch, no config, no expression
grammar. The single inward-facing dependency is `metrics.py`, which adapts
`RequestLifecycleMetric` records into rung measurements. All wiring into the
sweep lives in `load_generator.py` (`_probe_preprocess` and
`_run_probe_rung`).

## File map

| File | Contents |
| --- | --- |
| `result.py` | `RungResult`, `ProbeResult`, `BoundConstant`, `ConfidenceInterval`, `SaturationSignal`, `RESERVED_SYMBOLS` |
| `estimator.py` | Batch-means throughput, isotonic regression (PAVA), saturating-curve fit, `estimate_constants` (bootstrap CIs) |
| `ladder.py` | `LadderConfig`, `ConcurrencyLadder` (rung schedule and stop policy) |
| `stationarity.py` | CUSUM drift check over the measurement window |
| `metrics.py` | Adapter from request lifecycle metrics to `RungResult`, TTFT/ITL phase classification |

## Why closed loop

A closed loop keeps exactly N requests in flight: each completion admits the
next request, so the offered load adapts to whatever the server can serve.
Little's law (N = X * R) then holds for any arrival process and any backend
topology, which makes every rung self-checking: we measure throughput X and
latency R independently and report the relative residual of N = X * R
(`littles_law_residual`). A large residual means the measurement itself is
broken (clock skew, dispatch starvation, window misalignment), not that the
server is slow. Rungs also carry a CUSUM stationarity verdict, batch-means
standard errors for throughput, and an optional saturation signal
(`PREFILL_BOUND` / `DECODE_BOUND`) from TTFT/ITL inflation against the
first rung's unloaded baseline.

## How the ladder stops

Concurrency grows geometrically (`growth_factor`, default 2.0) from
`start_concurrency` until one of four sticky stop reasons:

- **plateau**: the relative throughput gain between the top two rungs is
  confidently below `gain_threshold`, using a one-sided upper confidence
  bound on the gain built from the batch-means standard errors. The SE
  awareness matters: a noisy rung must not fake a plateau.
- **client_bound**: the load generator is the bottleneck rather than the
  server. Each rung samples its own dispatch-loop lateness against the tick
  it intends to keep, and a typical (median) lateness above
  `CLIENT_LAG_FLOOR_SEC` and at least `inflation_threshold` times the unloaded
  rung's ends the ladder. Without this a generator that cannot feed the pool
  would report its own ceiling as the server's.
- **max_concurrency**: the configured or worker-capacity ceiling.
- **measurement_failed**: a rung produced no successful completion (every
  request failed, or the tail had to be cancelled). That rung cannot become a
  point on the curve, but the rungs below it did measure the server, so the
  ladder stops and the constants are estimated from those. The probe still
  fails when fewer than two usable rungs remain, because then there is no
  curve to fit.

After a plateau, one geometric-midpoint rung is optionally measured
(`refine`) to sharpen the knee estimate. A non-stationary rung is re-measured
once (`max_retries`) before being used as-is.

The stop reason is part of the result. Every reason except `PLATEAU` ends the
ladder before the curve flattens, so `ProbeResult.censored` is true for them
and the reported `r_sat` is labelled a lower bound; `failed_concurrency` names
the rung that produced no measurement when that is what ended the ladder.

## How constants are estimated

`estimate_constants` produces the reserved symbols `r_sat` (saturation
throughput) and `n_knee` (concurrency where isotonic throughput first
reaches `knee_fraction` of `r_sat`). Only rungs that `select_usable_rungs`
admits take part: stationary, not client-bound, Little's law residual within
`max_littles_law_residual`, and failed-request fraction within
`max_rung_error_rate`. A rung rejected on either of the last two would
otherwise enter the fit as a clean point that describes a broken measurement
or a failing backend. The estimator is deliberately two-layered:

1. Isotonic regression (pool-adjacent-violators) over the rungs gives a
   nonparametric, monotone throughput curve whose top value is the
   empirical plateau.
2. A saturating fit X(N) = mu * N / (K + N) supplies the asymptote, but is
   **accepted only when it lands within `fit_acceptance_ratio` (1.3x) of the
   empirical plateau**. If the ladder never got near saturation the fit
   extrapolates wildly, so the estimator falls back to the plateau, which is
   an honest lower bound.

Confidence intervals come from a parametric bootstrap that resamples each
rung's throughput from its batch-means standard error and re-runs the whole
estimate. The output is a dict of `BoundConstant`s: the measure-then-bind
contract under which a later phase may bind these symbols into sweep stage
rate expressions (#449, #563) without this package ever depending on the
expression grammar.

## Dispatch protocol (wiring, lives in load_generator.py)

The probe runs during sweep preprocessing, before stage generation:

- **Rungs use negative stage ids** (-1, -2, ...), which the report generator
  already filters out, so probe traffic never pollutes reports.
- **Concurrency is retuned through the workers' shared per-worker value.**
  Workers only read it between phases, and at process start the phase flag is
  already up, so the probe first parks all workers at a phase boundary
  (phase clear + barrier) before rung 1.
- **Every phase clear pairs a main-side `stage_barrier.wait()`** with one
  arrival per worker. This pairing is a hard protocol: a worker that
  observes the flag down proceeds to the barrier and, without a main-side
  wait, parks forever. The legacy burst path historically had no pairing and
  survived only because workers poll the flag on a ~0.5 s cadence and
  usually miss the microseconds-long window in which it is down; that race
  is now closed with an explicit pairing there too.
- **Rungs enqueue via chunked top-up** (high watermark `max(4N, 32)`) and
  end with a natural tail drain rather than `run_stage`'s enqueue-everything
  plus timeout-cancel, so no cancellations land inside a measurement window.
- The measurement window opens `settle_duration` after the rung starts and
  lasts `rung_duration`; metrics are collected mid-run through the request
  collector's `snapshot()` API.

## Forgone approaches, and why

- **Open-loop rate ramp** (step the request rate up until latency degrades).
  Above capacity an open-loop queue grows without bound, so every
  measurement taken there depends on how long you waited, and
  arrival-process variance confounds the capacity estimate. The closed loop
  self-regulates and measures a stable operating point per rung.
- **Burst drain-rate estimation** (the legacy sweep estimator this
  supersedes). A one-shot burst measures a transient: the drain-rate
  percentile mixes queue drain with steady-state capacity and carries no
  error bar. It remains the default; the probe is opt-in via `sweep.probe`.
- **Binary search on request rate against an SLO oracle.** Needs an SLO to
  define pass/fail, and sweep autoselection must work without one. It also
  yields a single point; the ladder yields the whole X(N) curve, from which
  an SLO-conditioned rate (`r_slo`) can later be derived.
- **Latency inflation as the stop rule.** TTFT/ITL inflation thresholds are
  model and hardware dependent, so inflation is demoted to a per-rung
  classification signal. The stop rule is throughput-gain based, which is
  scale free.
- **Parametric fit as the sole estimator.** Real serving stacks do not owe
  us a Monod-shaped curve (batching regimes, KV-cache pressure, and
  scheduler phase changes bend it). Hence the isotonic guardrail and the
  1.3x acceptance gate.
- **Reusing `run_stage` for rung dispatch.** Its enqueue-all plus
  timeout-cancel shape cancels stragglers, and cancellations inside the
  window would bias the rung. Hence the dedicated top-up loop.
- **A dedicated probe worker pool.** Rejected so the probe exercises exactly
  the dispatch path the real stages use, and to avoid new process machinery;
  retuning the existing workers' shared concurrency at phase boundaries is
  sufficient.
- **Report-schema integration of rung data.** Superseded: the run now emits
  `sweep_probe.json` from `ProbeResult.to_dict()`, and negative stage ids
  still keep the lifecycle reports untouched.
- **Binding constants into expressions in this change.** Deferred by design
  (measure-then-bind): this package emits `BoundConstant`s and reserves the
  symbol names; the grammar dependency stays out.

## Known limits

- The probe consumes items from the data generator, shared with the burst
  estimator: finite corpora (`total_count`) must be large enough to cover
  probe traffic or generation raises past the end.
- Closed-loop R(N) is a measurement device. It is not open-loop request
  latency and is never reported as such.
- The client-bound signal reads the generator's dispatch-loop lateness, so it
  catches a generator that cannot feed the pool. It does not catch a generator
  that is slow per request while still keeping its loop on time; for that,
  read the rung's Little's law residual and error rate, which also gate the
  fit.
- Phase classification requires streaming token timestamps; non-streaming
  runs degrade to `NONE` rather than guessing.
- `settle_duration` is a heuristic warm-in, not a detected steady-state
  onset; the CUSUM check catches drift that survives it, but only flags it.
