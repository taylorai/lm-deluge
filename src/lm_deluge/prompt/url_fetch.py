from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_url_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    request = Request(url, headers={"User-Agent": "lm-deluge"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Failed to fetch URL {url!r}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch URL {url!r}: {exc.reason}") from exc
