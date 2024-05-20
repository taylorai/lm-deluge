import tempfile
import json
import xxhash
from typing import Optional
from .api_requests.base import APIResponse

try:
    import plyvel
except ImportError:
    plyvel = None
    print("Warning: plyvel not installed, using in-memory cache.")

def hash_messages(messages: list[dict]) -> str:
    """
    Hash a list of messages.
    """
    hasher = xxhash.xxh64()
    hasher.update(json.dumps(messages).encode())
    return hasher.hexdigest()

def encode_api_response(response: APIResponse) -> bytes:
    """
    Encode an API response as a string.
    """
    return json.dumps(response.to_dict()).encode()

def decode_api_response(data: bytes) -> APIResponse:
    """
    Decode an API response from a string.
    """
    return APIResponse.from_dict(json.loads(data.decode()))

class Cache:
    """
    Store API responses based on their input messages.
    """
    def __init__(self, path: Optional[str] = None):
        if path is None:
            self.temp_file = tempfile.TemporaryFile(suffix=".db")
            path = self.temp_file.name
            print(f"Using temporary cache at {path}")
        else:
            self.temp_file = None
        self.path = path
        if plyvel is not None:
            self.db = plyvel.DB(path, create_if_missing=True)
        else:
            self.db = {}

    def get(self, messages: list[dict]) -> APIResponse:
        """
        Get an API response from the cache.
        """
        key = hash_messages(messages)
        if plyvel is not None:
            data = self.db.get(key.encode())
            if data is not None:
                return decode_api_response(data)
        else:
            data = self.db.get(key.encode()) # use the same key for in-memory cache
            if data is not None:
                return decode_api_response(data)
        return None
    
    def put(self, messages: list[dict], response: APIResponse):
        """
        Put an API response into the cache.
        """
        key = hash_messages(messages)
        if plyvel is not None:
            self.db.put(key.encode(), encode_api_response(response))
        else:
            self.db[key.encode()] = encode_api_response(response)

    def close(self):
        """
        Close the cache.
        """
        if plyvel is not None:
            self.db.close()
        else:
            self.db = {}
        if self.temp_file is not None:
            self.temp_file.close()