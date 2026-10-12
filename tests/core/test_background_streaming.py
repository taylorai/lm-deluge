"""Background Responses streaming, resume, cancellation, and polling regressions."""

import asyncio
import json
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import aiohttp

from lm_deluge import Conversation, TextDeltaEvent
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import OpenAIResponsesRequest
from lm_deluge.config import SamplingParams
from lm_deluge.tracker import StatusTracker


class Content:
    def __init__(self, events, failure=None, delay=0):
        self.events = events
        self.failure = failure
        self.delay = delay

    async def iter_any(self):
        for event in self.events:
            data = b"data: " + json.dumps(event).encode() + b"\n\n"
            # Exercise SSE framing across network chunks.
            for offset in range(0, len(data), 7):
                yield data[offset : offset + 7]
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure is not None:
            raise self.failure


class Response:
    status = 200
    headers: ClassVar[dict[str, str]] = {"Content-Type": "text/event-stream"}

    def __init__(self, events=(), *, data=None, failure=None, delay=0):
        self.content = Content(events, failure, delay)
        self.data = data
        if data is not None:
            self.headers = {"Content-Type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def raise_for_status(self):
        assert self.status == 200

    async def json(self):
        assert self.data is not None, "Attempted to parse SSE as JSON"
        return self.data

    async def text(self):
        return json.dumps(self.data)


class Session:
    def __init__(self, initial, resumes=()):
        self.initial = initial
        self.resumes = list(resumes)
        self.posts = []
        self.gets = []

    def post(self, **kwargs):
        self.posts.append(kwargs)
        if kwargs["url"].endswith("/cancel"):
            return Response(data={"status": "cancelled"})
        return self.initial

    def get(self, **kwargs):
        self.gets.append(kwargs)
        assert self.resumes, "Unexpected extra resume/poll"
        return self.resumes.pop(0)


def events():
    terminal = {
        "id": "resp1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Hello world", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    return [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp1", "status": "queued"},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "delta": "Hello ",
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "delta": "world",
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": terminal,
        },
    ]


def request(session, callback=None, timeout=30):
    context = RequestContext(
        task_id=1,
        model_name="gpt-4.1-mini",
        prompt=Conversation().user("hello"),
        sampling_params=SamplingParams(max_new_tokens=128),
        status_tracker=StatusTracker(1000, 500000, 10),
        background=True,
        use_responses_api=True,
        http_session=session,
        on_text_delta=callback,
        stream_round=2,
        request_timeout=timeout,
    )
    return OpenAIResponsesRequest(context)


async def happy_path():
    seen = []

    async def callback(event):
        seen.append(event)

    session = Session(Response(events()))
    result = await request(session, callback).execute_once()
    assert not result.is_error, result.error_message
    assert result.completion == "Hello world"
    assert result.raw_response == events()[-1]["response"]
    assert seen == [
        TextDeltaEvent("text", 2, "gpt-4.1-mini", "Hello "),
        TextDeltaEvent("text", 2, "gpt-4.1-mini", "world"),
    ]
    assert session.posts[0]["json"]["background"]
    assert session.posts[0]["json"]["stream"]
    assert not session.gets


async def resume():
    # EOF and transient read failures all reconnect with the same cursor. Even
    # if the server replays the last event, it must never reach the callback.
    for failure in (
        None,
        aiohttp.ClientPayloadError("truncated"),
        aiohttp.ClientConnectionError("disconnected"),
        asyncio.TimeoutError(),
    ):
        seen = []

        async def callback(event, seen=seen):
            seen.append(event.text)

        session = Session(
            Response(events()[:2], failure=failure),
            [Response(events()[1:])],
        )
        result = await request(session, callback).execute_once()
        assert not result.is_error, result.error_message
        assert result.completion == "Hello world"
        assert seen == ["Hello ", "world"]
        assert len(session.gets) == 1
        assert session.gets[0]["url"].endswith("/responses/resp1")
        assert session.gets[0]["params"] == {"stream": "true", "starting_after": 1}


async def failed():
    session = Session(
        Response(
            [
                events()[0],
                {
                    "type": "response.failed",
                    "sequence_number": 1,
                    "response": {
                        "id": "resp1",
                        "error": {"code": "invalid_prompt", "message": "bad prompt"},
                    },
                },
            ]
        )
    )

    async def callback(event):
        raise AssertionError("No text expected")

    result = await request(session, callback).execute_once()
    assert result.is_error and result.status_code == 400
    assert "bad prompt" in result.error_message
    assert not session.gets


async def callback_failure():
    failure = RuntimeError("callback stopped")

    async def callback(event):
        raise failure

    session = Session(Response(events()))
    try:
        await request(session, callback).execute_once()
    except RuntimeError as exc:
        assert exc is failure
    else:
        raise AssertionError("Callback failure was swallowed")
    assert session.posts[-1]["url"].endswith("/resp1/cancel")
    assert not session.gets


async def incomplete_and_missing_id():
    async def callback(event):
        pass

    terminal = events()[-1]["response"]
    terminal["status"] = "incomplete"
    terminal["incomplete_details"] = {"reason": "max_output_tokens"}
    session = Session(
        Response(
            [
                events()[0],
                {
                    "type": "response.incomplete",
                    "sequence_number": 1,
                    "response": terminal,
                },
            ]
        )
    )
    result = await request(session, callback).execute_once()
    assert result.is_error
    assert result.error_message == "Response incomplete: max_output_tokens"
    assert not session.gets

    for failure in (None, aiohttp.ClientPayloadError("truncated")):
        session = Session(Response(failure=failure))
        result = await request(session, callback).execute_once()
        assert result.is_error
        assert not session.gets
        assert len(session.posts) == 1  # No id to cancel or resume.


async def polling():
    session = Session(
        Response(data={"id": "resp1", "status": "queued"}),
        [
            Response(data={"id": "resp1", "status": "in_progress"}),
            Response(data=events()[-1]["response"]),
        ],
    )
    with patch("lm_deluge.api_requests.base.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await request(session).execute_once()
    assert not result.is_error, result.error_message
    assert result.completion == "Hello world"
    assert "stream" not in session.posts[0]["json"]
    assert len(session.gets) == 2 and sleep.await_count == 2
    assert all("params" not in call for call in session.gets)


async def limits():
    async def callback(event):
        pass

    session = Session(
        Response(events()[:2]), [Response(events()[:2]) for _ in range(3)]
    )
    result = await request(session, callback).execute_once()
    assert result.is_error and "3 resume attempts" in result.error_message
    assert len(session.gets) == 3
    assert session.posts[-1]["url"].endswith("/resp1/cancel")

    # Progress must reset the consecutive-attempt counter.
    session = Session(
        Response(events()[:2]),
        [Response([]), Response([]), Response(events()[2:3]), Response(events()[3:])],
    )
    result = await request(session, callback).execute_once()
    assert not result.is_error, result.error_message
    assert [call["params"]["starting_after"] for call in session.gets] == [1, 1, 1, 2]

    session = Session(Response(events()[:1], delay=1))
    result = await request(session, callback, timeout=0.02).execute_once()
    assert result.is_error
    assert result.error_message == "Request timed out (terminated by client)."
    assert session.posts[-1]["url"].endswith("/resp1/cancel")
    assert not session.gets


async def main():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-test-key"}):
        await happy_path()
        await resume()
        await failed()
        await callback_failure()
        await incomplete_and_missing_id()
        await polling()
        await limits()
    print(
        "PASS background streaming, resume deduplication, errors, cancellation, polling"
    )


if __name__ == "__main__":
    asyncio.run(main())
