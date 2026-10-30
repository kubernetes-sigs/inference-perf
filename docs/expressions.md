# Expressions

This is the reference for the expression grammar: what an expression can contain, how conditions work, and how value ranges are checked. Config docs that accept an expression link here instead of repeating it.

An expression is a string such as `"Normal(512, 128)"`, `"10 + 5*sin(2*pi*t/60)"` or `"t >= 60"`. It is parsed and validated once, when the config loads, so a bad one fails before the run starts.

Every checked table in this file is verified by `tests/required/docs/test_doc_examples.py`: each row must produce the value it claims, and each rejected example must be rejected. A wrong number here fails CI.

## Where expressions are accepted

| Field | `t` | Random | Rule |
| --- | --- | --- | --- |
| [`load.stages[].stop_condition`](./config.md#stop-conditions) (constant, poisson) | yes | no | A [condition](#conditions). |
| [`load.stages[].stop_condition`](./config.md#stop-conditions) (trace_session_replay) | yes | no | A [condition](#conditions) without `_over_sent` windows. |

## Grammar

| Element | Examples | Notes |
| --- | --- | --- |
| Numbers | `512`, `0.25`, `1/300` | A bare YAML number and the same number as a string mean the same thing. |
| Stage time | `t` | Seconds since the current stage started. Restarts at 0 in every stage. Only fields that vary over a stage allow it. |
| Arithmetic | `+ - * / **` | `2**(t/10)` is exponential growth. |
| Constants | `pi`, `E` | |
| Functions | `sin`, `exp`, `log`, `sqrt`, `Abs`, `Min`, `Max`, `Heaviside`, `Piecewise` | Any SymPy function. An unknown name is rejected. Conditions allow fewer (see [Conditions](#conditions)). |
| Random variables | `Normal(512, 128)`, `Poisson(64)` | See [Random variables](#random-variables). Only fields that allow randomness accept them. |
| Comparisons | `> >= < <=`, combined with `&` and `\|` | Conditions only. |

| Rejected expression | Why |
| --- | --- |
| `foo(3)` | `foo` is not a known function. |
| `x + 3` | `x` is not a known symbol. The only symbol is `t`. |
| `t >= 60` | A condition, not a number. Conditions go in fields that take one. |
| `1 +` | Does not parse. |

## Random variables

> [!IMPORTANT]
> The table below lists common distributions only. **Any constructor in SymPy's [`sympy.stats`](https://docs.sympy.org/latest/modules/stats.html) module works, about 75 in all, and SymPy's documentation is the reference for the full set and for every parameter.** Call a constructor without SymPy's leading name argument: `Normal(512, 128)`, not `Normal("x", 512, 128)`.
>
> Parameters follow SymPy's conventions, which do not always match the usual ones. Check SymPy's docs for any distribution not listed here.

| Expression | Mean |
| --- | --- |
| `Normal(512, 128)` | 512 |
| `Uniform(64, 256)` | 160 |
| `LogNormal(6, 0.5)` | 457 |
| `Exponential(1/300)` | 300 |
| `Poisson(64)` | 64 |
| `Gamma(2, 100)` | 200 |
| `Beta(2, 5)` | 0.2857 |
| `Pareto(100, 3)` | 150 |
| `128 + Poisson(64)` | 192 |
| `Min(LogNormal(6, 0.5), 4096)` | 457 |
| `Normal(512, 128) + Normal(0, 16)` | 512 |

What the parameters mean in the common cases:

- `Normal(mean, std)` and `Uniform(low, high)`: as usual.
- `LogNormal(mu, sigma)`: the mean and standard deviation of the underlying normal, not of the result. `LogNormal(6, 0.5)` has a mean near `exp(6 + 0.5**2/2)`, about 457.
- `Exponential(rate)`: a rate, so the mean is `1/rate`. Use `Exponential(1/300)` for a mean of 300.
- `Gamma(shape, scale)`: mean `shape * scale`.
- `Pareto(minimum, alpha)`: never below `minimum`; mean `alpha * minimum / (alpha - 1)`.

Each occurrence of a constructor is an independent draw, so `Normal(0, 1) + Normal(0, 1)` adds two separate samples.

Not supported yet:

- Constructors that take a list, such as `DiscreteUniform`: they can't be called from the grammar.
- `Triangular` and `Weibull`: accepted when the config loads, but they fail when sampled.

## Conditions

A condition is a comparison over `t`, used where a field waits for something to become true. It must be false at `t = 0` and then true from some time onward, and this is proved when the config loads, so a condition reduces to the exact time it becomes true. A condition can also read what the stage measures, see [Measured terms](#measured-terms).

| Condition | Holds from (s) |
| --- | --- |
| `t >= 60` | 60 |
| `t > 90.5` | 90.5 |
| `t >= 2*60` | 120 |
| `(t >= 300) \| (t >= 60)` | 60 |
| `(t >= 300) & (t >= 60)` | 300 |
| `60 < t` | 60 |
| `sqrt(t) >= 8` | 64 |
| `t**2 >= 3600` | 60 |
| `exp(t/10) > 100` | 46.05 |

A condition may use `+ * /`, powers, `sqrt`, `exp`, `log`, `Min`, `Max`, the four inequalities, `&` and `|`. Other functions are rejected because SymPy's solver gives wrong answers for some of them (`Abs` among them). `&` and `|` bind tighter than a comparison, so parenthesise each comparison.

| Rejected condition | Why |
| --- | --- |
| `t = 60` | Equality holds at a single instant, which a check can step over. |
| `t == 60` | Same. |
| `t < 60` | True at the start, then false. It must become true and stay true. |
| `(t >= 60) & (t < 120)` | Only true between 60 and 120. |
| `t >= 0` | Already true at `t = 0`. |
| `sin(t) > 0` | `sin` is not allowed in a condition. |
| `Abs(t - 30) > 10` | `Abs` is not allowed in a condition. |
| `t >= 60 & t < 120` | `&` binds tighter than `>=`; parenthesise each comparison. |
| `t` | A number, not a condition. |
| `Normal(0, 1) >= t` | Random. A condition must be the same on every run. |
| `requests >= 1000` | `requests` is not a known symbol. |

## Measured terms

A condition can read what the stage measures, so it can become true on a latency or error threshold as well as on time. A measured term is a window function over finished requests, written in PromQL's range syntax: `quantile_over_time(0.99, ttft[30s])` is the p99 time to first token of the requests that finished in the last 30 seconds. Terms are evaluated in-process on every finished request, from the same per-request values the report is computed from, so a quantile is the exact quantile of those requests rather than a histogram estimate.

> **Series names are short, not the report's field names.** A condition says `ttft` where the report says `time_to_first_token`, and `latency` where it says `request_latency`. Writing the report's name is rejected with the short name in the error. The table below is the full list. It is checked against the code in both directions, so a series cannot be added, renamed or dropped without this table changing too.

| Series | Report field | Unit | Samples |
| --- | --- | --- | --- |
| `latency` | `request_latency` | s | Every successful request: end minus start. |
| `ttft` | `time_to_first_token` | s | Successful streamed requests. |
| `tpot` | `time_per_output_token` | s | Successful streamed requests with more than one output token. |
| `ntpot` | `normalized_time_per_output_token` | s | Successful requests with output tokens. |
| `schedule_delay` | `schedule_delay` | s | Every request: how late it was sent against its schedule. |
| `error` | none | 0 or 1 | Every request: 1 if it failed. |

There is no inter-token latency series. The report re-derives each token's timestamp from the raw response chunks when it writes the report, so a value measured during the run would be a different number under the same name.

A window function is `<aggregate>_over_<window>(...)`, and any aggregate works with any window. Like the series table, the two tables below are checked against the code in both directions.

| Aggregate | Reads |
| --- | --- |
| `avg` | Mean of the samples. |
| `min` | Smallest sample. |
| `max` | Largest sample. |
| `sum` | Sum of the samples. |
| `count` | Number of samples. |
| `stddev` | Population standard deviation, as PromQL computes it. |
| `quantile` | The q-quantile, with q first and between 0 and 1, interpolated linearly as PromQL does. |

There is no median: use `quantile(0.5, ...)`.

| Window | Selects |
| --- | --- |
| `_over_time` | PromQL's window: samples that finished inside a range such as ttft[30s], with optional offset and @. |
| `_over_last` | Ours: the last n samples, written (series, n). |
| `_over_sent` | Ours: requests a up to b in the order they were sent, counting from 0, written (series, a, b). |

| Window function | Reads |
| --- | --- |
| `quantile_over_time(0.99, ttft[30s])` | p99 TTFT of requests that finished in the last 30 s. |
| `avg_over_time(latency[10s] offset 10s)` | Mean latency of requests that finished in the 10 s before the last 10 s. |
| `sum_over_time(error[60s] @ 120)` | Failures that finished between stage seconds 60 and 120. |
| `avg_over_last(latency, 100)` | Mean latency of the last 100 successful requests. |
| `quantile_over_last(0.9, ttft, 50)` | p90 TTFT of the last 50 streamed requests. |
| `avg_over_sent(ttft, 10, 30)` | Mean TTFT of the 11th to 30th request sent. |

A range is a PromQL duration: `500ms`, `30s`, `5m`, `1h`, `1m30s`. `offset` moves the range back. `@` ends the range at a stage second, the same clock as `t`. That differs from PromQL, where `@` takes a unix timestamp. Subqueries (`[30s:5s]`) are not supported.

A term has no value until its window is complete, and a comparison on a term with no value is false, so the rest of the condition decides. A time range is complete once it lies inside the stage, so `ttft[30s]` has no value before `t = 30`. `_over_last(series, n)` needs `n` samples. `_over_sent(series, a, b)` needs every request from `a` up to `b` to finish, and a failed request counts as finished even though it has no latency. A window with no samples has no value either.

A condition can also read these signals directly, with no window.

| Signal | Reads |
| --- | --- |
| `completed` | Requests finished so far in the stage, successes and failures. |
| `errors` | Failed requests so far in the stage. |
| `in_flight` | Requests sent and not yet finished. |
| `since_last_completion` | Seconds since a request last finished, or since the stage started if none has. |

A measured term is the system under test, so it cannot be solved like `t`. The time part of the condition is still proved when the config loads, twice. With every measured comparison false the condition must still become true on time alone: that is the time it holds by. With every measured comparison true it must again become true from some time onward: that is the earliest a measured term can make it hold. So a measured term can only make a condition hold sooner, never keep it from holding.

| Condition | Holds by (s) | Can hold from (s) |
| --- | --- | --- |
| `(t >= 600) \| (quantile_over_time(0.99, ttft[30s]) > 2)` | 600 | 0 |
| `((t >= 60) & (avg_over_last(latency, 100) > 1)) \| (t >= 600)` | 600 | 60 |
| `(t >= 300) \| (errors >= 10)` | 300 | 0 |
| `(t >= 300) \| (avg_over_last(error, 100) > 0.05)` | 300 | 0 |
| `(t >= 900) \| (avg_over_time(latency[10s]) > 1.5*avg_over_time(latency[10s] offset 10s))` | 900 | 0 |
| `(t >= 900) \| (avg_over_last(ttft, 20) > 2*avg_over_sent(ttft, 10, 30))` | 900 | 0 |
| `(t >= 600) \| ((in_flight > 0) & (since_last_completion > 30))` | 600 | 0 |

| Rejected condition | Why |
| --- | --- |
| `quantile_over_time(0.99, ttft[30s]) > 2` | Never holds on time alone. Write `(t >= 600) \| (...)`. |
| `(t >= 60) & (avg_over_last(ttft, 50) > 2)` | Same: with the term false it never becomes true. |
| `(t >= 600) \| (ttft > 2)` | A series is not a value. Read it through a window function, such as `avg_over_last(ttft, 100)`. |
| `(t >= 600) \| (avg_over_time(time_to_first_token[30s]) > 2)` | The report's field name. The series is `ttft`. |
| `(t >= 600) \| (median_over_time(ttft[30s]) > 2)` | No median. Use `quantile_over_time(0.5, ttft[30s])`. |
| `(t >= 600) \| (avg_over_time(ttft[30s:5s]) > 2)` | Subqueries are not supported. |
| `(t >= 600) \| (avg_over_time(ttft[10s] @ start()) > 2)` | `@` takes a stage second. |
| `(t >= 600) or (avg_over_last(ttft, 50) > 2)` | Combine comparisons with `\|` and `&`, not `or` and `and`. |

## Value ranges

A field can restrict the values an expression may take, for example a rate must not be negative. A value that can provably fall outside the range is rejected when the config loads; one that can only be caught while sampling raises then. Fields that would rather truncate random draws say so in their own docs.

Some fields need an expression's largest possible value, for example to check that a request fits the model's context. That bound is worked out from the expression: each random variable contributes its support, and `Min`, `Max`, `Abs` and arithmetic combine them. The bound can be wider than the true range but never narrower. An expression with no finite upper bound, such as a bare `Normal(50, 10)`, can be capped with `Min(...)`.

| Expression | Lower | Upper |
| --- | --- | --- |
| `Uniform(64, 256)` | 64 | 256 |
| `Min(LogNormal(6, 0.5), 4096)` | 0 | 4096 |
| `Min(2*Uniform(10, 20), 30)` | 20 | 30 |
| `128 + Poisson(64)` | 128 | inf |
| `Normal(50, 10)` | -inf | inf |
| `Min(Normal(50, 10), 99)` | -inf | 99 |
| `Beta(2, 5)` | 0 | 1 |
| `Abs(Normal(0, 1))` | 0 | inf |
