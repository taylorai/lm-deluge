import csv
import json
from pathlib import Path

from lm_deluge.prompt import Conversation

CSV_PATH = Path("/Users/benjamin/Downloads/assistant_messages.csv")


def test_csv_conversations_load_with_unknown_provider():
    if not CSV_PATH.exists():
        print("assistant_messages.csv not available on this machine")
        return
    with CSV_PATH.open() as handle:
        reader = csv.DictReader(handle)
        errors: list[tuple[int, Exception]] = []
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            messages_raw = row.get("messages")
            if not messages_raw:
                continue
            payload = json.loads(messages_raw)
            try:
                msgs, provider = Conversation.from_unknown(payload)
                print("✅ Parsed row", row_count, f"({provider})")
            except Exception as exc:  # pragma: no cover - defensive reporting
                errors.append((row_count - 1, exc))
                break

    assert row_count > 0, "CSV contained no rows"
    assert not errors, f"Failed to parse rows: {errors}"


if __name__ == "__main__":
    test_csv_conversations_load_with_unknown_provider()
