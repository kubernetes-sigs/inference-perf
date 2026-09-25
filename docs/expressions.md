# Expressions

This is the reference for the expression grammar: what an expression can contain, how conditions work, and how value ranges are checked. Config docs that accept an expression link here instead of repeating it.

An expression is a string such as `"Normal(512, 128)"`, `"10 + 5*sin(2*pi*t/60)"` or `"t >= 60"`. It is parsed and validated once, when the config loads, so a bad one fails before the run starts.

Every checked table in this file is verified by `tests/required/docs/test_doc_examples.py`: each row must produce the value it claims, and each rejected example must be rejected. A wrong number here fails CI.

## Where expressions are accepted

| Field | `t` | Random | Rule |
| --- | --- | --- | --- |
| [`load.stages[].rate`](./config.md#rate-expressions) (constant, poisson) | yes | no | Nonnegative over the stage and not zero everywhere. |
| [`load.stages[].stop_condition`](./config.md#stop-conditions) (constant, poisson) | yes | no | A [condition](#conditions). |
| [`data.input_distribution`, `data.output_distribution`](./config.md#data-generation) | no | yes | Rounded to whole tokens. |
| [`data.shared_prefix.system_prompt_len`, `question_len`, `output_len`](./config.md#data-generation) | no | yes | Rounded to whole tokens. |
| [`data.conversation_replay` knobs](./conversation_replay.md#configuration-guide) | no | yes | Counts rounded; `tool_call_latency_sec` keeps fractions. |
| [`data.synthetic_agentic` knobs and `context_compaction`](./synthetic_agentic.md) | no | yes | Counts rounded; latencies keep fractions. |
| [`data.multimodal.*.count`](./config.md#multimodal-data-generation) | no | yes | Rounded to whole items. |
| [`data.multimodal.*.insertion_point`](./config.md#multimodal-data-generation) | no | yes | Provably within [0, 1]. |

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

A condition is a comparison over `t`, used where a field needs something to become true, such as a stage's stop condition. It must be false at `t = 0` and then true from some time onward, and this is proved when the config loads, so a condition reduces to the exact time it becomes true.

| Condition | Ends at (s) |
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
