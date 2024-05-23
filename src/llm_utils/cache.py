import tempfile
import json
import sqlite3
import xxhash
from typing import Optional, Any
from .api_requests.base import APIResponse

try:
    import plyvel
except ImportError:
    plyvel = None
    print("Warning: plyvel not installed, cannot use LevelDB.")

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

class DistributedDictCache:
    """
    Use distributed dictionary (e.g. Modal Dict) as a cache.
    Pass in the dictionary object to use. Cache must implement
    'get' and 'put' methods.
    """
    def __init__(self, cache: Any):
        self.cache = cache

    def get(self, messages: list[dict]) -> APIResponse:
        """
        Get an API response from the cache.
        """
        data = self.cache.get(hash_messages(messages))
        if data is not None:
            return decode_api_response(data)
        return None
    
    def put(self, messages: list[dict], response: APIResponse):
        """
        Put an API response into the cache.
        """
        self.cache.put(
            hash_messages(messages),
            encode_api_response(response)
        )

class LevelDBCache:
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
           raise ImportError("plyvel not installed, cannot use LevelDBCache.")

    def get(self, messages: list[dict]) -> APIResponse:
        """
        Get an API response from the cache.
        """
        key = hash_messages(messages)
        data = self.db.get(key.encode())
        if data is not None:
            return decode_api_response(data)
        return None
    
    def put(self, messages: list[dict], response: APIResponse):
        """
        Put an API response into the cache.
        """
        key = hash_messages(messages)
        self.db.put(key.encode(), encode_api_response(response))

    def close(self):
        """
        Close the cache.
        """
        self.db.close()
        if self.temp_file is not None:
            self.temp_file.close()

class SqliteCache:
    """
    Same interface as LevelDBCache, but uses SQLite as KV store instead.
    Good to use on systems where LevelDB installation is problematic.
    """
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)"
        )
        self.conn.commit()

    def get(self, messages: list[dict]) -> APIResponse:
        """
        Get an API response from the cache.
        """
        key = hash_messages(messages)
        self.cursor.execute("SELECT value FROM cache WHERE key=?", (key,))
        data = self.cursor.fetchone()
        if data is not None and len(data) > 0:
            return decode_api_response(data[0])
        return None
    
    def put(self, messages: list[dict], response: APIResponse):
        """
        Put an API response into the cache.
        """
        key = hash_messages(messages)
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, encode_api_response(response))
        )
        self.conn.commit()

    def close(self):
        """
        Close the cache.
        """
        self.conn.close()