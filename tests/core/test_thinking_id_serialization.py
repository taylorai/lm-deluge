"""Test Thinking id preservation through log serialization."""

from lm_deluge.prompt import Message, Thinking


def test_thinking_id_round_trip():
    thinking = Thinking(content="reasoning", id="rs_test123")
    msg = Message("assistant", [thinking])
    log = msg.to_log()
    round_trip = Message.from_log(log)

    thinking_parts = [p for p in round_trip.parts if isinstance(p, Thinking)]
    assert len(thinking_parts) == 1
    assert thinking_parts[0].id == "rs_test123"
    print("✓ Thinking id preserved through Message log round-trip")


if __name__ == "__main__":
    test_thinking_id_round_trip()
