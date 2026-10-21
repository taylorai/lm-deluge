"""Live tests for Cloudflare's Clef decision models.

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (Workers Paid plan or
prepaid AI Gateway credits).

Run directly from the repository root:

    .venv/bin/python tests/one_off/test_cloudflare_clef_live.py
"""

import asyncio
import base64
import io
import json
import math
import os
import struct
import wave

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.prompt import Image

MODEL_IDS = ["clef-cf", "clef-omni-cf", "clef-flash-cf"]

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments, invoices, and refunds",
            "technical": "Outages, errors, and configuration",
            "sales": "Plans and upgrades",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"],
    },
}


def _require_credentials() -> None:
    missing = [
        name
        for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing credentials required by this live test: " + ", ".join(missing)
        )


def _tone_wav_data_url(seconds: float = 1.0, freq: float = 440.0) -> str:
    rate = 16000
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(rate * seconds))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:audio/wav;base64,{encoded}"


async def _run(
    model_id: str, prompt: Conversation, extra_body: dict
) -> tuple[APIResponse, dict]:
    client = LLMClient(
        model_names=model_id,
        max_attempts=1,
        request_timeout=180,
        extra_body=extra_body,
    )
    try:
        response = await client.start(prompt)
    finally:
        client.close()

    assert not response.is_error, (
        f"{model_id} returned an error: {response.error_message}"
    )
    assert response.completion, f"{model_id} returned no completion"
    return response, json.loads(response.completion)


def _check_probabilities(probabilities: dict, keys: set[str]) -> None:
    assert set(probabilities) == keys, probabilities
    assert abs(sum(probabilities.values()) - 1) < 1e-3, probabilities


async def check_text_decisions(model_id: str) -> str:
    response, answers = await _run(
        model_id,
        Conversation().user(
            "Checkout has been failing for every customer for the last hour."
        ),
        {"questions": QUESTIONS},
    )
    assert set(answers) == set(QUESTIONS), answers

    urgent = answers["urgent"]
    assert urgent["type"] == "noul" and 0 <= urgent["noul"] <= 1, urgent
    assert urgent["noul"] > 0.5, urgent

    team = answers["team"]
    assert team["type"] == "choice", team
    _check_probabilities(team["probabilities"], {"billing", "technical", "sales"})
    assert team["choice"] == "technical", team

    severity = answers["severity"]
    assert severity["type"] == "score", severity
    _check_probabilities(severity["probabilities"], {"0", "1", "2", "3"})
    assert 0 <= severity["score"] <= 3, severity

    assert response.usage is not None and response.usage.input_tokens > 0
    assert response.cost is not None and response.cost > 0
    return (
        f"urgent={urgent['noul']:.2f} team={team['choice']} "
        f"severity={severity['score']:.2f} cost=${response.cost:.6f}"
    )


async def check_image_decision(model_id: str) -> str:
    _, answers = await _run(
        model_id,
        Conversation().user(
            "A user uploaded this photo.", image=Image("tests/image.jpg")
        ),
        {
            "questions": {
                "animal": {
                    "type": "noul",
                    "instructions": "Does the image show an animal?",
                }
            }
        },
    )
    animal = answers["animal"]
    assert animal["type"] == "noul", animal
    assert animal["noul"] > 0.5, animal
    return f"animal={animal['noul']:.2f}"


async def check_audio_decision(model_id: str) -> str:
    _, answers = await _run(
        model_id,
        Conversation().user("An audio clip was recorded."),
        {
            "audio": [_tone_wav_data_url()],
            "questions": {
                "kind": {
                    "type": "choice",
                    "instructions": "What does the audio clip contain?",
                    "criteria": {
                        "speech": "A person talking",
                        "tone": "A steady pure tone or beep",
                        "music": "Music with melody or rhythm",
                    },
                }
            },
        },
    )
    kind = answers["kind"]
    assert kind["type"] == "choice", kind
    _check_probabilities(kind["probabilities"], {"speech", "tone", "music"})
    return f"kind={kind['choice']} probs={kind['probabilities']}"


async def test_clef_models_live() -> None:
    _require_credentials()
    checks = [(model_id, check_text_decisions) for model_id in MODEL_IDS]
    checks += [(model_id, check_image_decision) for model_id in MODEL_IDS]
    checks.append(("clef-omni-cf", check_audio_decision))

    failures: list[str] = []
    for model_id, check in checks:
        label = f"{model_id} {check.__name__}"
        print(f"Testing {label}...")
        try:
            summary = await check(model_id)
        except Exception as exc:  # noqa: BLE001 - run every check before failing
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  FAIL: {failures[-1]}")
        else:
            print(f"  PASS: {summary}")

    assert not failures, "Clef live failures:\n- " + "\n- ".join(failures)


if __name__ == "__main__":
    asyncio.run(test_clef_models_live())
    print("All Clef live checks passed!")
