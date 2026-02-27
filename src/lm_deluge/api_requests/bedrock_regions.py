import json
import os
import random
import threading
import time
from functools import lru_cache
from typing import Any

from ..models import APIModel

DEFAULT_BEDROCK_REGION_COOLDOWN_SECONDS = 10.0
MAX_BEDROCK_REGION_COOLDOWN_SECONDS = 120.0
UNSUPPORTED_BEDROCK_REGION_COOLDOWN_SECONDS = 60.0 * 60.0

_BEDROCK_REGION_STATE_LOCK = threading.Lock()
_BEDROCK_REGION_ROUND_ROBIN_INDEX: dict[str, int] = {}
_BEDROCK_REGION_COOLDOWN_UNTIL: dict[tuple[str, str], float] = {}


def _safe_positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _fallback_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"


@lru_cache(maxsize=1)
def _parse_region_weight_overrides() -> dict[str, dict[str, int]]:
    """Load model-specific region weight overrides from environment.

    Expected env var format:
      DELUGE_BEDROCK_REGION_WEIGHTS_JSON='{
        "claude-4.6-sonnet-bedrock": {"us-east-1": 3, "us-west-2": 1},
        "global.anthropic.claude-sonnet-4-6": {"us-east-1": 10, "eu-west-1": 4}
      }'

    Keys can be either the model registry ID or the underlying Bedrock profile name.
    """
    raw = os.getenv("DELUGE_BEDROCK_REGION_WEIGHTS_JSON", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(parsed, dict):
        return {}

    normalized: dict[str, dict[str, int]] = {}
    for model_key, region_map in parsed.items():
        if not isinstance(model_key, str) or not isinstance(region_map, dict):
            continue

        cleaned: dict[str, int] = {}
        for region, weight in region_map.items():
            if not isinstance(region, str):
                continue
            normalized_weight = _safe_positive_int(weight, default=0)
            if normalized_weight > 0:
                cleaned[region] = normalized_weight

        if cleaned:
            normalized[model_key] = cleaned

    return normalized


def _override_region_weights(model: APIModel) -> dict[str, int] | None:
    overrides = _parse_region_weight_overrides()
    return overrides.get(model.id) or overrides.get(model.name)


def _effective_region_weights(model: APIModel) -> dict[str, int] | None:
    override = _override_region_weights(model)
    if override is not None:
        return override
    if isinstance(model.regions, dict):
        return {
            region: _safe_positive_int(weight, default=0)
            for region, weight in model.regions.items()
            if _safe_positive_int(weight, default=0) > 0
        }
    return None


def _dedupe_preserve_order(regions: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for region in regions:
        if region in seen:
            continue
        seen.add(region)
        ordered.append(region)
    return ordered


def configured_bedrock_regions(model: APIModel) -> list[str]:
    effective_weights = _effective_region_weights(model)
    if effective_weights is not None:
        regions = list(effective_weights.keys())
    elif isinstance(model.regions, list):
        regions = _dedupe_preserve_order(model.regions)
    else:
        regions = []

    if regions:
        return regions
    return [_fallback_region()]


def bedrock_region_count(model: APIModel) -> int:
    return len(configured_bedrock_regions(model))


def _region_cooldown_until(model_id: str, region: str) -> float:
    with _BEDROCK_REGION_STATE_LOCK:
        return _BEDROCK_REGION_COOLDOWN_UNTIL.get((model_id, region), 0.0)


def _region_is_available(model_id: str, region: str, now: float) -> bool:
    return _region_cooldown_until(model_id, region) <= now


def has_available_bedrock_source_regions(model: APIModel) -> bool:
    now = time.monotonic()
    return any(
        _region_is_available(model.id, region, now)
        for region in configured_bedrock_regions(model)
    )


def _pick_round_robin(model_id: str, regions: list[str]) -> str:
    with _BEDROCK_REGION_STATE_LOCK:
        index = _BEDROCK_REGION_ROUND_ROBIN_INDEX.get(model_id, 0)
        region = regions[index % len(regions)]
        _BEDROCK_REGION_ROUND_ROBIN_INDEX[model_id] = index + 1
    return region


def _pick_weighted(
    regions: list[str],
    region_weights: dict[str, int],
) -> str:
    weights = [
        _safe_positive_int(region_weights.get(region, 1), default=1)
        for region in regions
    ]
    return random.choices(regions, weights=weights, k=1)[0]


def _pick_soonest_cooldown_region(model_id: str, regions: list[str]) -> str:
    with _BEDROCK_REGION_STATE_LOCK:
        return min(
            regions,
            key=lambda region: _BEDROCK_REGION_COOLDOWN_UNTIL.get(
                (model_id, region), 0.0
            ),
        )


def pick_bedrock_source_region(model: APIModel) -> str:
    """Pick a source region for a Bedrock request.

    Lists are treated as round-robin. Dicts are treated as weighted spray.
    Regions can be temporarily cooled down due to rate limits.
    """

    regions = configured_bedrock_regions(model)
    region_weights = _effective_region_weights(model)
    now = time.monotonic()
    available = [
        region for region in regions if _region_is_available(model.id, region, now)
    ]

    if region_weights is not None:
        if available:
            return _pick_weighted(available, region_weights)
        return _pick_soonest_cooldown_region(model.id, regions)

    if available:
        return _pick_round_robin(model.id, available)
    return _pick_soonest_cooldown_region(model.id, regions)


def _set_region_cooldown_until(
    model_id: str, region: str, cooldown_until: float
) -> None:
    with _BEDROCK_REGION_STATE_LOCK:
        key = (model_id, region)
        current = _BEDROCK_REGION_COOLDOWN_UNTIL.get(key, 0.0)
        if cooldown_until > current:
            _BEDROCK_REGION_COOLDOWN_UNTIL[key] = cooldown_until


def mark_bedrock_region_rate_limited(
    model: APIModel, region: str, retry_after: float | None
) -> None:
    if retry_after is None:
        delay = DEFAULT_BEDROCK_REGION_COOLDOWN_SECONDS
    else:
        delay = max(0.0, min(retry_after, MAX_BEDROCK_REGION_COOLDOWN_SECONDS))
    _set_region_cooldown_until(model.id, region, time.monotonic() + delay)


def mark_bedrock_region_unsupported(model: APIModel, region: str) -> None:
    _set_region_cooldown_until(
        model.id,
        region,
        time.monotonic() + UNSUPPORTED_BEDROCK_REGION_COOLDOWN_SECONDS,
    )


def is_probably_region_scoped_bedrock_error(error_message: str | None) -> bool:
    if not error_message:
        return False

    message = error_message.lower()

    # Some Bedrock source regions can reject otherwise-valid credentials with
    # this generic auth message. Treat as region-scoped so we rotate regions.
    if "security token included in the request is invalid" in message:
        return True

    if "region" not in message:
        return False

    region_markers = (
        "source region",
        "from region",
        "in this region",
        "requested region",
    )
    unsupported_markers = (
        "not supported",
        "unsupported",
        "not available",
        "cannot be found",
        "can't be found",
        "not found",
        "not authorized",
        "access denied",
    )

    return any(marker in message for marker in region_markers) and any(
        marker in message for marker in unsupported_markers
    )


def reset_bedrock_region_state_for_tests() -> None:
    with _BEDROCK_REGION_STATE_LOCK:
        _BEDROCK_REGION_ROUND_ROBIN_INDEX.clear()
        _BEDROCK_REGION_COOLDOWN_UNTIL.clear()
    _parse_region_weight_overrides.cache_clear()
