# Expressions and Distributions Guide

Inference Perf supports dynamic, expression-based configurations for load generation and data distribution. This allows you to define complex, time-varying workloads and realistic data sizes using mathematical expressions and statistical distributions.

We use [SymPy](https://www.sympy.org/) for parsing and evaluating these expressions.

---

## Deterministic Expressions

Deterministic expressions are used where stability and predictability are required, such as in `concurrency_level` for `ConcurrentLoadStage`. These expressions can vary over time but cannot contain random variables.

### The Time Variable `t`

In time-varying expressions, you can use the variable `t`, which represents the **number of seconds elapsed since the start of the current stage**.

### Supported Functions

You can use standard mathematical operators (`+`, `-`, `*`, `/`, `**`) and functions, including:
-   `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`
-   `Min(a, b, ...)`: Returns the minimum value.
-   `Max(a, b, ...)`: Returns the maximum value.
-   `Piecewise((val1, cond1), (val2, cond2), ...)`: Defines step functions or conditional logic.

### Examples

#### 1. Ramping Load (Linear Increase)
To linearly increase concurrency over time, you can use a simple linear function of `t`.

```yaml
load:
  type: concurrent
  stages:
  - num_requests: 1000
    # Starts at 10 and increases by 1 every 2 seconds
    concurrency_level: "10 + 0.5 * t"
```

#### 2. Cyclical Load (Waves)
To simulate wave patterns (e.g., daily or hourly fluctuations), use trigonometric functions like `sin` or `cos`.

```yaml
load:
  type: concurrent
  stages:
  - num_requests: 5000
    # Fluctuates between 30 and 70 with a period of ~62.8 seconds
    concurrency_level: "50 + 20 * sin(t / 10)"
```

#### 3. Load Spikes (Step Functions)
To simulate sudden spikes or steps in load, use `Piecewise`.

```yaml
load:
  type: concurrent
  stages:
  - num_requests: 3000
    # 10 for first 30s, spikes to 100 for next 10s, then returns to 10
    concurrency_level: "Piecewise((10, t < 30), (100, t < 40), (10, True))"
```

#### 4. Clamping (Min/Max Bounds)
To ensure that an expression stays within safe boundaries, combine `Min` and `Max`.

```yaml
load:
  type: concurrent
  stages:
  - num_requests: 2000
    # Increases with time but never goes below 10 or above 50
    concurrency_level: "Min(50, Max(10, 10 + t))"
```

---

## Stochastic Distributions

Stochastic distributions allow you to introduce randomness. They are supported for:
-   `interval` in `StandardLoadStage` (to vary request arrival times).
-   `input_distribution` and `output_distribution` in `DataConfig` (for `random` and `synthetic` datasets).
-   `question_distribution` and `output_distribution` in `SharedPrefix` configuration.

> [!IMPORTANT]
> Distributions are **NOT** allowed in `concurrency_level` to prevent unstable worker pool resizing.

### Supported Distributions

We support all continuous random variables available in `sympy.stats`. Some common ones include:

-   `Normal(mean, std_dev)`
-   `Poisson(lambda)`
-   `Exponential(rate)`
-   `Uniform(min, max)`

### Examples

#### 1. Poisson Arrival Process
A Poisson process has Exponentially distributed intervals. You can specify this directly for the stage interval.

```yaml
load:
  type: constant
  stages:
  - duration: 60
    # Request intervals are exponentially distributed with rate 10 (mean interval 0.1s)
    interval: "Exponential(10)"
```

#### 2. Random Data Sizes
You can use expressions to define distributions for input and output lengths in `data` configuration.

```yaml
data:
  type: random
  # Prompt lengths are normally distributed around 512
  input_distribution: "Normal(512, 100)"
  # Output lengths are normally distributed around 256, clamped between 10 and 1024
  output_distribution: "Min(1024, Max(10, Normal(256, 50)))"
```

#### 3. Shared Prefix Distributions
Expressions can also be used in `shared_prefix` configurations to define distributions for questions and outputs within groups.

```yaml
data:
  type: shared_prefix
  shared_prefix:
    num_groups: 5
    num_prompts_per_group: 10
    system_prompt_len: 100
    question_len: 50
    output_len: 50
    # Question lengths are normally distributed
    question_distribution: "Normal(50, 5)"
    # Output lengths are exponentially distributed
    output_distribution: "Exponential(1/50)"
```

#### 3. Time-Varying Rates with Randomness
You can combine time-varying deterministic expressions with random distributions for intervals in `StandardLoadStage`.

```yaml
load:
  type: constant
  stages:
  - duration: 60
    # Rate increases over time, so interval (1/rate) decreases
    interval: "Exponential(10 + t/6)"
```
