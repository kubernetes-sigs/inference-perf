from collections.abc import Generator
from datetime import datetime
import time
import jmespath.exceptions
from pydantic import BaseModel, ValidationError
import pytest

from inference_perf.circuit_breaker import _initialized_circuit_breakers, get_circuit_breaker, init_circuit_breakers
from inference_perf.circuit_breaker.simple_breaker import SimpleCircuitBreaker
from inference_perf.circuit_breaker.triggers import HitSample
from inference_perf.circuit_breaker.triggers.consecutive import Consecutive
from inference_perf.circuit_breaker.triggers.rate_over_window import RateOverWindow
from inference_perf.config.circuit_breaker import CircuitBreakerConfig, TriggerConsecutive, TriggerRateOverWindow


class DummyMetric(BaseModel):
    foo: str = "bar"
    val: int = 10
    end_time: float = 0.0


@pytest.fixture(autouse=True)
def clear_breakers() -> Generator[None, None, None]:
    _initialized_circuit_breakers.clear()
    yield
    _initialized_circuit_breakers.clear()


def test_init_circuit_breakers() -> None:
    config = CircuitBreakerConfig(
        name="test", metrics={"matches": ["foo == 'bar'"]}, triggers=[TriggerConsecutive(type="consecutive", threshold=3)]
    )
    init_circuit_breakers([config])

    cb = get_circuit_breaker("test")
    assert isinstance(cb, SimpleCircuitBreaker)

    with pytest.raises(RuntimeError, match="Circuit breakers already initialized"):
        init_circuit_breakers([config])

    with pytest.raises(ValueError, match="Unknown circuit breaker: unknown"):
        get_circuit_breaker("unknown")


def test_simple_breaker_match_rule_semantics() -> None:
    config = CircuitBreakerConfig(
        name="test",
        metrics={"matches": ["foo == 'bar'", "foo == 'baz'"], "rules": ["val > `5`", "val < `0`"]},
        triggers=[TriggerConsecutive(type="consecutive", threshold=1)],
    )
    cb = SimpleCircuitBreaker(config)

    # Matches first expr, rule matches first expr
    cb.feed(DummyMetric(foo="bar", val=10, end_time=time.time()))
    assert cb.is_open()
    cb.reset()
    assert not cb.is_open()

    # Matches second expr, rule matches second expr
    cb.feed(DummyMetric(foo="baz", val=-5, end_time=time.time()))
    assert cb.is_open()
    cb.reset()

    # No match
    cb.feed(DummyMetric(foo="qux", val=10, end_time=time.time()))
    assert not cb.is_open()

    # Match, but no rule hit
    cb.feed(DummyMetric(foo="bar", val=2, end_time=time.time()))
    assert not cb.is_open()


def test_simple_breaker_invalid_expression() -> None:
    config = CircuitBreakerConfig(
        name="test", metrics={"matches": ["foo =="]}, triggers=[TriggerConsecutive(type="consecutive", threshold=1)]
    )
    with pytest.raises(jmespath.exceptions.JMESPathError):
        SimpleCircuitBreaker(config)


def test_consecutive_trigger() -> None:
    t = Consecutive(threshold=3)
    ts = datetime.now()

    t.update(HitSample(ts, 1))
    assert not t.fired()

    t.update(HitSample(ts, 1))
    assert not t.fired()

    t.update(HitSample(ts, 1))
    assert t.fired()  # threshold boundary

    t.reset()
    assert not t.fired()

    # reset to 0 on a miss
    t.update(HitSample(ts, 1))
    t.update(HitSample(ts, 1))
    t.update(HitSample(ts, 0))  # miss
    t.update(HitSample(ts, 1))
    t.update(HitSample(ts, 1))
    assert not t.fired()
    t.update(HitSample(ts, 1))
    assert t.fired()


def test_rate_over_window_trigger() -> None:
    # min_samples gating
    t = RateOverWindow(window_sec=10.0, threshold=0.5, min_samples=3)
    ts1 = datetime.fromtimestamp(100)

    t.update(HitSample(ts1, 1))
    t.update(HitSample(ts1, 1))
    assert not t.fired()  # only 2 samples

    t.update(HitSample(ts1, 0))
    assert t.fired()  # 3 samples, 2 hits => rate=0.66 > 0.5

    t.reset()
    assert not t.fired()

    # threshold=0.0 edge
    t_zero = RateOverWindow(window_sec=10.0, threshold=0.0, min_samples=0)
    t_zero.update(HitSample(ts1, 0))
    assert t_zero.fired()

    # window eviction
    t_evict = RateOverWindow(window_sec=5.0, threshold=0.5, min_samples=0)
    t_evict.update(HitSample(datetime.fromtimestamp(100), 1))
    t_evict.update(HitSample(datetime.fromtimestamp(101), 1))
    assert t_evict.fired()

    t_evict.reset()
    t_evict.update(HitSample(datetime.fromtimestamp(100), 1))
    t_evict.update(HitSample(datetime.fromtimestamp(101), 1))
    # Add a miss at 106, which should evict 100
    t_evict.update(HitSample(datetime.fromtimestamp(106), 0))
    # window has (101: hit), (106: miss) -> 1/2 = 0.5
    assert t_evict.fired()

    t_evict.reset()
    t_evict.update(HitSample(datetime.fromtimestamp(100), 1))
    t_evict.update(HitSample(datetime.fromtimestamp(101), 1))
    # Add misses at 107, evicting both hits
    t_evict.update(HitSample(datetime.fromtimestamp(107), 0))
    assert not t_evict.fired()


def test_config_validation() -> None:
    with pytest.raises(ValidationError):
        TriggerConsecutive(type="consecutive", threshold=0)  # ge=1

    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=-1.0, threshold=0.5)

    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=1.0, threshold=1.5)
