# import sqlite3
# import xxhash
# import json
# from dataclasses import dataclass

# @dataclass
# class SqliteCache:
#     """
#     Cache class to avoid re-computing completions on the same prompt.
#     Use InstructStore or MessageStore if you want to just save inputs/outputs without hashing
#     (e.g. to store data for fine-tuning or inspection).
#     """
#     path: str

#     def __post_init__(self):
#         self.conn = sqlite3.connect(self.path)
#         self.cursor = self.conn.cursor()
#         self.cursor.execute(
#             "CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, content TEXT)"
#         )
#         self.conn.commit()

#     @staticmethod
#     def get_hash(messages):
#         hasher = xxhash.xxh64()
#         hasher.update(json.dumps(messages).encode())
#         hash_key = hasher.hexdigest()
#         return hash_key

#     def get_from_cache(self, messages):
#         hash_key = self.get_hash(messages)
#         self.cursor.execute("SELECT content FROM cache WHERE hash=?", (hash_key,))
#         return self.cursor.fetchone()

#     def set_to_cache(self, messages, content):
#         hash_key = self.get_hash(messages)
#         try:
#             self.cursor.execute(
#                 "INSERT INTO cache (hash, content) VALUES (?, ?)", (hash_key, content)
#             )
#         # if failed due to unique constraint, update instead
#         except sqlite3.IntegrityError:
#             self.cursor.execute(
#                 "UPDATE cache SET content=? WHERE hash=?", (content, hash_key)
#             )
#         except Exception as e:
#             print(f"Error setting cache: {e}")
#         self.conn.commit()

# @dataclass
# class InstructStore:
#     """
#     Store class to save (input_text, output_text) without hashing.
#     Use SqliteCache if you want to cache completions based on the prompt.
#     """
#     path: str

#     def __post_init__(self):
#         self.conn = sqlite3.connect(self.path)
#         self.cursor = self.conn.cursor()
#         self.cursor.execute(
#             "CREATE TABLE IF NOT EXISTS store (id INTEGER PRIMARY KEY, inputs TEXT, outputs TEXT)"
#         )
#         self.conn.commit()

#     def get_from_store(self, inputs):
#         self.cursor.execute("SELECT outputs FROM store WHERE inputs=?", (inputs,))
#         return self.cursor.fetchone()

#     def set_to_store(self, inputs, outputs):
#         try:
#             self.cursor.execute(
#                 "INSERT INTO store (inputs, outputs) VALUES (?, ?)", (inputs, outputs)
#             )
#         # if failed due to unique constraint, update instead
#         except sqlite3.IntegrityError:
#             self.cursor.execute(
#                 "UPDATE store SET outputs=? WHERE inputs=?", (outputs, inputs)
#             )
#         except Exception as e:
#             print(f"Error setting store: {e}")
#         self.conn.commit()

# @dataclass
# class MessageStore:
#     """
#     Store class to save (input_messages, output_text) without hashing.
#     Use SqliteCache if you want to cache completions based on the prompt.
#     """
