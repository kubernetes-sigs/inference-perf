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

from inference_perf.loadgen.load_timer import ExpressionLoadTimer


def test_expression_load_timer_float() -> None:
    timer = ExpressionLoadTimer(interval=0.1, duration=2.0)
    gen = timer.start_timer()
    t1 = next(gen)
    t2 = next(gen)
    assert t2 >= t1  # Should be monotonically increasing
    # For constant interval 0.1, difference should be exactly 0.1
    assert abs((t2 - t1) - 0.1) < 1e-5


def test_expression_load_timer_poisson_expr() -> None:
    # Poisson process with rate 10 means interval is Exponential(10)
    timer = ExpressionLoadTimer(interval="Exponential(10)", duration=2.0)
    gen = timer.start_timer()
    t1 = next(gen)
    t2 = next(gen)
    assert t2 >= t1  # Should be monotonically increasing


def test_expression_load_timer_time_varying() -> None:
    # Time varying interval
    timer = ExpressionLoadTimer(interval="0.1 + t/10", duration=2.0)
    gen = timer.start_timer()
    t1 = next(gen)
    t2 = next(gen)
    assert t2 >= t1
