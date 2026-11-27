# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Property tests: what Expression proves about a value holds for every draw."""

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from inference_perf.utils.numeric.expression import Expression

_DRAWS = 2000
_SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

_small = st.integers(min_value=0, max_value=50)
# dyadic floats are exact in binary64, so numpy sampling cannot drift past sympy's proven bounds by 1 ULP
_SCALES = st.one_of(
    st.integers(min_value=-4, max_value=4).filter(lambda k: k != 0).map(str),
    st.sampled_from(["-2.5", "-1.5", "-0.5", "-0.25", "0.25", "0.5", "1.5"]),
)
_DISCRETE = st.one_of(
    st.builds(lambda lam: f"Poisson({lam})", st.integers(min_value=1, max_value=40)),
    st.builds(lambda k: f"Geometric(1/{k})", st.integers(min_value=2, max_value=10)),
    st.builds(
        lambda n, k: f"Binomial({n}, 1/{k})", st.integers(min_value=1, max_value=50), st.integers(min_value=2, max_value=5)
    ),
)
_CONTINUOUS = st.one_of(
    st.builds(lambda lo, width: f"Uniform({lo}, {lo + width})", _small, st.integers(min_value=1, max_value=100)),
    st.builds(lambda mu, sigma: f"Normal({mu}, {sigma})", _small, st.integers(min_value=1, max_value=20)),
    st.builds(lambda mu: f"LogNormal({mu}, 1/2)", st.integers(min_value=0, max_value=5)),
    st.builds(lambda k: f"Exponential(1/{k})", st.integers(min_value=1, max_value=20)),
)


# grows a leaf into a tree with +, -, signed integer/float scaling, Abs, Min/Max caps and the roundings
def _extend(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.one_of(
        st.builds(lambda a, b: f"({a}) + ({b})", children, children),
        st.builds(lambda a, b: f"({a}) - ({b})", children, children),
        st.builds(lambda a, k: f"({k})*({a})", children, _SCALES),
        st.builds(lambda a: f"Abs({a})", children),
        st.builds(lambda a, c: f"Min({a}, {c})", children, st.integers(min_value=1, max_value=4000)),
        st.builds(lambda a, c: f"Max({a}, {c})", children, _small),
        st.builds(lambda a, c: f"Min({a}, {c}/2)", children, st.integers(min_value=1, max_value=200)),
        st.builds(lambda a: f"floor({a})", children),
        st.builds(lambda a: f"ceiling({a})", children),
        st.builds(lambda a: f"floor({a} + 1/2)", children),
    )


_EXPRESSIONS = st.recursive(st.one_of(_DISCRETE, _CONTINUOUS, _small.map(str)), _extend, max_leaves=4)


# 2000 seeded float64 draws of raw, e.g. 'Poisson(4) - Uniform(0, 10)' with seed 0.
def _draws(raw: str, seed: int) -> "np.typing.NDArray[np.float64]":
    return np.asarray(Expression(raw, allow_time=False).sample(rng=np.random.default_rng(seed), size=_DRAWS), dtype=np.float64)


# Any expression from the grammar, e.g. 'Min(floor(Normal(20, 5)) - (-0.5)*(Poisson(3)), 37)'. Expected: bounds
# always prove (never silently None) and all 2000 draws lie inside them. A bound can be loose, never wrong.
@_SETTINGS
@given(raw=_EXPRESSIONS, seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_draws_stay_inside_proven_bounds(raw: str, seed: int) -> None:
    bounds = Expression(raw, allow_time=False).bounds
    assert bounds is not None, f"{raw}: expected provable bounds, got None"
    draws = _draws(raw, seed)
    assert bounds[0] <= draws.min() and draws.max() <= bounds[1], (
        f"{raw}: bounds {bounds}, drew [{draws.min()}, {draws.max()}]"
    )


# Any expression proven integer-valued, e.g. 'Poisson(4) - floor((-0.5)*(Uniform(0, 10)))'. Expected: every draw
# is a whole number, and assume() fails the run if is_integer_valued ever collapses to False.
@_SETTINGS
@given(raw=_EXPRESSIONS, seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_integer_valued_expressions_draw_whole_numbers(raw: str, seed: int) -> None:
    assume(Expression(raw, allow_time=False).is_integer_valued)
    draws = _draws(raw, seed)
    assert np.all(draws == np.floor(draws)), f"{raw}: drew a fraction, e.g. {draws[draws != np.floor(draws)][:3]}"
