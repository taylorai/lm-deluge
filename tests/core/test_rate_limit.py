"""Tests for rate limit cooldown logic and retry-after header parsing."""

import time
from unittest.mock import MagicMock

from lm_deluge.api_requests.base import parse_retry_after
from lm_deluge.tracker import (
    DEFAULT_COOLDOWN_SECONDS,
    MAX_COOLDOWN_SECONDS,
    StatusTracker,
)


def _make_response(headers: dict[str, str]) -> MagicMock:
    """Create a mock aiohttp ClientResponse with given headers."""
    resp = MagicMock()
    resp.headers = headers
    return resp


# --- parse_retry_after tests ---


def test_parse_retry_after_ms():
    resp = _make_response({"retry-after-ms": "5000"})
    assert parse_retry_after(resp) == 5.0


def test_parse_retry_after_ms_fractional():
    resp = _make_response({"retry-after-ms": "1500"})
    assert parse_retry_after(resp) == 1.5


def test_parse_retry_after_seconds():
    resp = _make_response({"retry-after": "21"})
    assert parse_retry_after(resp) == 21.0


def test_parse_retry_after_seconds_float():
    resp = _make_response({"retry-after": "10.5"})
    assert parse_retry_after(resp) == 10.5


def test_parse_retry_after_ms_takes_priority():
    resp = _make_response({"retry-after-ms": "3000", "retry-after": "10"})
    assert parse_retry_after(resp) == 3.0


def test_parse_retry_after_missing():
    resp = _make_response({})
    assert parse_retry_after(resp) is None


def test_parse_retry_after_capped():
    resp = _make_response({"retry-after": "999"})
    assert parse_retry_after(resp) == MAX_COOLDOWN_SECONDS


def test_parse_retry_after_negative_clamped():
    resp = _make_response({"retry-after": "-5"})
    assert parse_retry_after(resp) == 0.0


def test_parse_retry_after_invalid():
    resp = _make_response({"retry-after": "not-a-number"})
    assert parse_retry_after(resp) is None


# --- StatusTracker cooldown tests ---


def _make_tracker() -> StatusTracker:
    return StatusTracker(
        max_requests_per_minute=100,
        max_tokens_per_minute=100_000,
        max_concurrent_requests=10,
    )


def test_cooldown_default():
    t = _make_tracker()
    t.rate_limit_exceeded()
    pause = t.seconds_to_pause
    assert DEFAULT_COOLDOWN_SECONDS - 1 < pause <= DEFAULT_COOLDOWN_SECONDS


def test_cooldown_with_retry_after():
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=25.0)
    pause = t.seconds_to_pause
    assert 24 < pause <= 25


def test_cooldown_bump_forward():
    """If we're mid-cooldown and get a longer retry-after, deadline extends."""
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=10.0)
    # Simulate some time passing
    t._cooldown_until -= 5  # pretend 5s have passed
    remaining_before = t.seconds_to_pause  # ~5s left
    # Now a new 429 says wait 20s
    t.rate_limit_exceeded(retry_after=20.0)
    remaining_after = t.seconds_to_pause
    assert remaining_after > remaining_before
    assert 19 < remaining_after <= 20


def test_cooldown_no_bump_backward():
    """A shorter retry-after shouldn't shorten an existing longer cooldown."""
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=30.0)
    pause_after_first = t.seconds_to_pause
    # A shorter one comes in — should NOT reduce
    t.rate_limit_exceeded(retry_after=5.0)
    pause_after_second = t.seconds_to_pause
    # Should be nearly identical (just a tiny bit of time passing)
    assert pause_after_second >= pause_after_first - 0.1


def test_cooldown_print_id_increments_on_bump():
    t = _make_tracker()
    pid0 = t._cooldown_print_id
    t.rate_limit_exceeded(retry_after=10.0)
    pid1 = t._cooldown_print_id
    assert pid1 > pid0
    # A second call that bumps forward should increment again
    t.rate_limit_exceeded(retry_after=20.0)
    pid2 = t._cooldown_print_id
    assert pid2 > pid1


def test_cooldown_print_id_no_increment_if_no_bump():
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=30.0)
    pid1 = t._cooldown_print_id
    # A shorter one should NOT increment (no bump)
    t.rate_limit_exceeded(retry_after=5.0)
    pid2 = t._cooldown_print_id
    assert pid2 == pid1


def test_cooldown_expires():
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=1.0)
    # Simulate expiry
    t._cooldown_until = time.time() - 1
    assert t.seconds_to_pause == 0.0


def test_rate_limit_error_count():
    t = _make_tracker()
    t.rate_limit_exceeded()
    t.rate_limit_exceeded()
    t.rate_limit_exceeded()
    assert t.num_rate_limit_errors == 3


def test_retry_after_capped_at_max():
    t = _make_tracker()
    t.rate_limit_exceeded(retry_after=999.0)
    assert t.seconds_to_pause <= MAX_COOLDOWN_SECONDS


if __name__ == "__main__":
    test_parse_retry_after_ms()
    test_parse_retry_after_ms_fractional()
    test_parse_retry_after_seconds()
    test_parse_retry_after_seconds_float()
    test_parse_retry_after_ms_takes_priority()
    test_parse_retry_after_missing()
    test_parse_retry_after_capped()
    test_parse_retry_after_negative_clamped()
    test_parse_retry_after_invalid()
    test_cooldown_default()
    test_cooldown_with_retry_after()
    test_cooldown_bump_forward()
    test_cooldown_no_bump_backward()
    test_cooldown_print_id_increments_on_bump()
    test_cooldown_print_id_no_increment_if_no_bump()
    test_cooldown_expires()
    test_rate_limit_error_count()
    test_retry_after_capped_at_max()
    print("All rate limit tests passed!")
