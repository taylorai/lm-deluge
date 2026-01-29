"""Simple SSE (Server-Sent Events) decoder."""

from dataclasses import dataclass


@dataclass
class SSEvent:
    """A Server-Sent Event."""

    event: str = "message"
    data: str = ""
    id: str = ""
    retry: int | None = None


class SSEDecoder:
    """
    Parse Server-Sent Events from a text stream.

    Usage:
        decoder = SSEDecoder()
        for line in lines:
            event = decoder.decode_line(line)
            if event:
                # Process event
    """

    def __init__(self):
        self._event = ""
        self._data: list[str] = []
        self._last_event_id = ""
        self._retry: int | None = None

    def decode_line(self, line: str) -> SSEvent | None:
        """
        Process a single line from the SSE stream.

        Returns an SSEvent when a complete event is ready (signaled by empty line),
        or None if more data is needed.
        """
        # Strip trailing CR if present (handles both \n and \r\n)
        line = line.rstrip("\r\n")

        if not line:
            # Empty line = dispatch event
            if not self._event and not self._data:
                return None

            event = SSEvent(
                event=self._event or "message",
                data="\n".join(self._data),
                id=self._last_event_id,
                retry=self._retry,
            )

            # Reset for next event (but keep last_event_id per SSE spec)
            self._event = ""
            self._data = []
            self._retry = None

            return event

        # Comment line
        if line.startswith(":"):
            return None

        # Parse field:value
        if ":" in line:
            field_name, value = line.split(":", 1)
            # Remove single leading space from value if present
            if value.startswith(" "):
                value = value[1:]
        else:
            field_name = line
            value = ""

        if field_name == "event":
            self._event = value
        elif field_name == "data":
            self._data.append(value)
        elif field_name == "id":
            # Ignore IDs containing null
            if "\0" not in value:
                self._last_event_id = value
        elif field_name == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass
        # Unknown fields are ignored per spec

        return None
