import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sqlite import SqliteManager

dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_people_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
        )
        conn.execute(
            "INSERT INTO people (name, age) VALUES (?, ?)",
            ("alice", 30),
        )
        conn.execute(
            "INSERT INTO people (name, age) VALUES (?, ?)",
            ("bob", 41),
        )
        conn.commit()
    finally:
        conn.close()


def _create_library_db(path: Path) -> None:
    """Create a richer DB with multiple tables for live agent tests."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, birth_year INTEGER)"
        )
        conn.execute(
            "CREATE TABLE books ("
            "  id INTEGER PRIMARY KEY,"
            "  title TEXT NOT NULL,"
            "  author_id INTEGER REFERENCES authors(id),"
            "  year INTEGER,"
            "  genre TEXT"
            ")"
        )
        conn.executemany(
            "INSERT INTO authors (name, birth_year) VALUES (?, ?)",
            [
                ("George Orwell", 1903),
                ("Aldous Huxley", 1894),
                ("Ray Bradbury", 1920),
            ],
        )
        conn.executemany(
            "INSERT INTO books (title, author_id, year, genre) VALUES (?, ?, ?, ?)",
            [
                ("1984", 1, 1949, "dystopia"),
                ("Animal Farm", 1, 1945, "allegory"),
                ("Brave New World", 2, 1932, "dystopia"),
                ("Fahrenheit 451", 3, 1953, "dystopia"),
                ("The Martian Chronicles", 3, 1950, "sci-fi"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — existing + expanded
# ---------------------------------------------------------------------------


def test_sqlite_manager_on_existing_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path, tool_name="sql")
        tool = manager.get_tools()[0]
        assert isinstance(tool.run, Callable)

        list_payload = json.loads(tool.run(command="list_tables"))
        assert list_payload["ok"]
        objects = list_payload["result"]["objects"]
        assert any(obj["name"] == "people" for obj in objects)

        desc_payload = json.loads(
            tool.run(command="describe_table", table_name="people")
        )
        assert desc_payload["ok"]
        columns = desc_payload["result"]["columns"]
        assert any(col["name"] == "id" and col["type"] == "INTEGER" for col in columns)

        query_payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT id, name FROM people ORDER BY id",
            )
        )
        assert query_payload["ok"]
        assert query_payload["result"]["columns"] == ["id", "name"]
        assert query_payload["result"]["rows"][1]["name"] == "bob"


def test_query_output_formats_csv_and_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        csv_payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT id, name FROM people ORDER BY id",
                output_format="csv",
            )
        )
        assert csv_payload["ok"]
        csv_output = csv_payload["result"]["output"]
        assert "id,name" in csv_output
        assert "1,alice" in csv_output

        yaml_payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT id, name FROM people ORDER BY id",
                output_format="yaml",
            )
        )
        assert yaml_payload["ok"]
        yaml_output = yaml_payload["result"]["output"]
        assert "- id: 1" in yaml_output
        assert "name: alice" in yaml_output


def test_read_only_mode_allows_complex_cte_without_heuristics():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path, read_only=True)
        tool = manager.get_tools()[0]

        payload = json.loads(
            tool.run(
                command="query",
                sql=(
                    "WITH prep AS ("
                    "  SELECT id, name || ' update token' AS label FROM people"
                    ") "
                    "SELECT id, label FROM prep ORDER BY id"
                ),
            )
        )
        assert payload["ok"]
        assert payload["result"]["rows"][0]["label"] == "alice update token"


def test_get_tools_exclude_blocks_query():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools(exclude={"query"})[0]

        blocked_payload = json.loads(tool.run(command="query", sql="SELECT 1"))
        assert blocked_payload["ok"] is False
        assert "disabled" in blocked_payload["message"]


def test_from_dicts_type_inference_and_nulls():
    rows = [
        {"id": 1, "age": 30, "meta": {"city": "NYC"}},
        {"id": "2", "meta": {"city": "SF"}},
        {"id": 3, "age": None, "meta": {"city": "LA"}, "nickname": "Ace"},
    ]

    manager = SqliteManager.from_dicts(rows, table_name="people")
    try:
        tool = manager.get_tools()[0]

        desc_payload = json.loads(
            tool.run(command="describe_table", table_name="people")
        )
        assert desc_payload["ok"]

        type_map = {
            column["name"]: column["type"]
            for column in desc_payload["result"]["columns"]
        }
        assert type_map["id"] == "TEXT"
        assert type_map["age"] == "INTEGER"
        assert type_map["meta"] == "JSONB"
        assert type_map["nickname"] == "TEXT"

        query_payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT id, age, nickname FROM people WHERE id = '2'",
            )
        )
        assert query_payload["ok"]
        assert query_payload["result"]["rows"][0]["age"] is None
        assert query_payload["result"]["rows"][0]["nickname"] is None
    finally:
        temp_db = manager.db_path
        manager.close()
        assert not temp_db.exists()


def test_from_dicts_rejects_nested_and_scalar_mix():
    rows = [{"value": {"x": 1}}, {"value": "oops"}]
    try:
        SqliteManager.from_dicts(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for nested/scalar mixed column")


def test_from_dicts_saves_to_disk():
    """from_dicts with an explicit db_path writes a persistent file and doesn't delete it on close."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "saved.db"
        rows = [
            {"product": "apple", "price": 1.5},
            {"product": "banana", "price": 0.75},
        ]
        manager = SqliteManager.from_dicts(rows, db_path=db_path, table_name="products")
        try:
            assert manager.db_path == db_path
            tool = manager.get_tools()[0]

            query_payload = json.loads(
                tool.run(
                    command="query",
                    sql="SELECT product, price FROM products ORDER BY price",
                )
            )
            assert query_payload["ok"]
            assert query_payload["result"]["rows"][0]["product"] == "banana"
            assert query_payload["result"]["rows"][1]["price"] == 1.5
        finally:
            manager.close()
        # File was explicitly placed on disk — should NOT be deleted on close.
        assert db_path.exists()


def test_from_dicts_float_inference():
    rows = [
        {"score": 9.5, "label": "A"},
        {"score": 7.0, "label": "B"},
        {"score": None, "label": "C"},
    ]
    manager = SqliteManager.from_dicts(rows, table_name="grades")
    try:
        tool = manager.get_tools()[0]
        desc = json.loads(tool.run(command="describe_table", table_name="grades"))
        assert desc["ok"]
        type_map = {col["name"]: col["type"] for col in desc["result"]["columns"]}
        assert type_map["score"] == "REAL"
        assert type_map["label"] == "TEXT"

        q = json.loads(
            tool.run(command="query", sql="SELECT score FROM grades WHERE label = 'C'")
        )
        assert q["ok"]
        assert q["result"]["rows"][0]["score"] is None
    finally:
        manager.close()


def test_from_dicts_nested_json_round_trip():
    rows = [
        {"name": "Alice", "tags": ["python", "data"]},
        {"name": "Bob", "tags": ["go", "systems"]},
    ]
    manager = SqliteManager.from_dicts(rows, table_name="users")
    try:
        tool = manager.get_tools()[0]
        q = json.loads(
            tool.run(command="query", sql="SELECT name, tags FROM users ORDER BY name")
        )
        assert q["ok"]
        # tags is stored as JSON string; sqlite returns it as a string
        alice_tags = q["result"]["rows"][0]["tags"]
        assert "python" in alice_tags
        assert "data" in alice_tags
    finally:
        manager.close()


def test_describe_table_with_sample_rows():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        desc = json.loads(
            tool.run(command="describe_table", table_name="people", sample_rows=2)
        )
        assert desc["ok"]
        assert len(desc["result"]["sample_rows"]) == 2
        assert desc["result"]["sample_rows"][0]["name"] == "alice"


def test_list_tables_with_row_counts():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        payload = json.loads(tool.run(command="list_tables", include_row_counts=True))
        assert payload["ok"]
        obj = next(o for o in payload["result"]["objects"] if o["name"] == "people")
        assert obj["row_count"] == 2


def test_query_with_bound_parameters():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT name FROM people WHERE age > ?",
                query_params=[35],
            )
        )
        assert payload["ok"]
        assert len(payload["result"]["rows"]) == 1
        assert payload["result"]["rows"][0]["name"] == "bob"


def test_query_max_rows_truncation():
    rows = [{"n": i} for i in range(50)]
    manager = SqliteManager.from_dicts(rows, table_name="nums")
    try:
        tool = manager.get_tools()[0]
        payload = json.loads(
            tool.run(command="query", sql="SELECT n FROM nums", max_rows=10)
        )
        assert payload["ok"]
        assert payload["result"]["row_count"] == 10
        assert payload["result"]["truncated"] is True
    finally:
        manager.close()


def test_from_dicts_empty_rejects():
    try:
        SqliteManager.from_dicts([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty rows")


def test_tool_cache_same_tool_object():
    """The Tool object inside the returned list is the same instance on repeated calls."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tools_a = manager.get_tools()
        tools_b = manager.get_tools()
        # The internal cache returns the same Tool object (same identity)
        assert tools_a[0] is tools_b[0]


def test_describe_nonexistent_table_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        payload = json.loads(
            tool.run(command="describe_table", table_name="nonexistent")
        )
        assert payload["ok"] is False
        assert "not found" in payload["message"].lower()


def test_missing_sql_for_query_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        payload = json.loads(tool.run(command="query"))
        assert payload["ok"] is False


def test_tsv_output_format():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "people.db"
        _create_people_db(db_path)

        manager = SqliteManager(db_path)
        tool = manager.get_tools()[0]

        payload = json.loads(
            tool.run(
                command="query",
                sql="SELECT id, name FROM people ORDER BY id",
                output_format="tsv",
            )
        )
        assert payload["ok"]
        output = payload["result"]["output"]
        assert "\t" in output
        lines = output.strip().split("\n")
        assert lines[0] == "id\tname"
        assert lines[1] == "1\talice"


def test_from_dicts_writable_allows_insert():
    rows = [{"val": 1}, {"val": 2}]
    manager = SqliteManager.from_dicts(rows, table_name="nums", read_only=False)
    try:
        tool = manager.get_tools()[0]
        insert_payload = json.loads(
            tool.run(
                command="query",
                sql="INSERT INTO nums (val) VALUES (99)",
                allow_write=True,
            )
        )
        assert insert_payload["ok"]

        q = json.loads(tool.run(command="query", sql="SELECT COUNT(*) AS c FROM nums"))
        assert q["ok"]
        assert q["result"]["rows"][0]["c"] == 3
    finally:
        manager.close()


def test_manager_raises_for_nonexistent_path():
    try:
        SqliteManager("/tmp/this_db_should_not_exist_xyz_abc.db")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")


# ---------------------------------------------------------------------------
# Live network tests — LLM uses the sqlite tool
# ---------------------------------------------------------------------------


async def test_live_agent_queries_from_dicts():
    """LLM uses sqlite tool built from a list-of-dicts to answer a question."""
    rows = [
        {"city": "New York", "population_m": 8.3, "country": "USA"},
        {"city": "Tokyo", "population_m": 13.9, "country": "Japan"},
        {"city": "Paris", "population_m": 2.1, "country": "France"},
        {"city": "Mumbai", "population_m": 12.5, "country": "India"},
        {"city": "Sydney", "population_m": 5.3, "country": "Australia"},
    ]
    manager = SqliteManager.from_dicts(rows, table_name="cities")
    try:
        tools = manager.get_tools()
        llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=512)
        conv = Conversation().user(
            "Using the sqlite tool, find the city with the highest population. "
            "Reply with just the city name."
        )
        final_conv, response = await llm.run_agent_loop(conv, tools=tools, max_rounds=5)
        assert response is not None
        assert not response.is_error, response.error_message
        assert "Tokyo" in (
            response.completion or ""
        ), f"Expected 'Tokyo' in response, got: {response.completion!r}"
        print(f"  Agent answer: {response.completion!r}")
    finally:
        manager.close()


async def test_live_agent_queries_db_on_disk():
    """LLM uses sqlite tool backed by a DB file created on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "library.db"
        _create_library_db(db_path)

        manager = SqliteManager(db_path)
        tools = manager.get_tools()

        llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=512)
        conv = Conversation().user(
            "Using the sqlite tool, list all tables in the database, then tell me "
            "how many books were written by Ray Bradbury. "
            "Reply with just a number."
        )
        final_conv, response = await llm.run_agent_loop(conv, tools=tools, max_rounds=6)
        assert response is not None
        assert not response.is_error, response.error_message
        completion = response.completion or ""
        assert "2" in completion, f"Expected '2' in response, got: {completion!r}"
        print(f"  Agent answer: {completion!r}")


async def test_live_agent_schema_discovery():
    """LLM uses describe_table to understand schema before querying."""
    rows = [
        {"employee": "Alice", "dept": "Engineering", "salary": 120000},
        {"employee": "Bob", "dept": "Marketing", "salary": 95000},
        {"employee": "Carol", "dept": "Engineering", "salary": 130000},
        {"employee": "Dave", "dept": "Marketing", "salary": 88000},
    ]
    manager = SqliteManager.from_dicts(rows, table_name="employees")
    try:
        tools = manager.get_tools()
        llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=512)
        conv = Conversation().user(
            "Using the sqlite tool, first describe the 'employees' table, then "
            "calculate the average salary for the Engineering department. "
            "Reply with just the number."
        )
        final_conv, response = await llm.run_agent_loop(conv, tools=tools, max_rounds=6)
        assert response is not None
        assert not response.is_error, response.error_message
        completion = response.completion or ""
        # avg of 120000 and 130000 is 125000
        assert (
            "125000" in completion or "125,000" in completion
        ), f"Expected '125000' in response, got: {completion!r}"
        print(f"  Agent answer: {completion!r}")
    finally:
        manager.close()


async def test_live_agent_from_dicts_on_disk_path():
    """LLM queries a DB created via from_dicts with an explicit on-disk path."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "explicit.db"
        rows = [
            {"lang": "Python", "year": 1991, "paradigm": "multi"},
            {"lang": "Rust", "year": 2010, "paradigm": "systems"},
            {"lang": "Go", "year": 2009, "paradigm": "systems"},
            {"lang": "JavaScript", "year": 1995, "paradigm": "multi"},
        ]
        manager = SqliteManager.from_dicts(
            rows, db_path=db_path, table_name="languages"
        )
        try:
            tools = manager.get_tools()
            llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=512)
            conv = Conversation().user(
                "Using the sqlite tool, find the oldest programming language "
                "(smallest year). Reply with just the language name."
            )
            final_conv, response = await llm.run_agent_loop(
                conv, tools=tools, max_rounds=5
            )
            assert response is not None
            assert not response.is_error, response.error_message
            assert "Python" in (
                response.completion or ""
            ), f"Expected 'Python' in response, got: {response.completion!r}"
            print(f"  Agent answer: {response.completion!r}")
        finally:
            manager.close()
        assert db_path.exists(), "Explicit db_path should persist after close()"


async def main():
    print("=" * 60)
    print("SqliteManager unit tests")
    print("=" * 60)

    test_sqlite_manager_on_existing_db()
    print("✓ test_sqlite_manager_on_existing_db")

    test_query_output_formats_csv_and_yaml()
    print("✓ test_query_output_formats_csv_and_yaml")

    test_read_only_mode_allows_complex_cte_without_heuristics()
    print("✓ test_read_only_mode_allows_complex_cte_without_heuristics")

    test_get_tools_exclude_blocks_query()
    print("✓ test_get_tools_exclude_blocks_query")

    test_from_dicts_type_inference_and_nulls()
    print("✓ test_from_dicts_type_inference_and_nulls")

    test_from_dicts_rejects_nested_and_scalar_mix()
    print("✓ test_from_dicts_rejects_nested_and_scalar_mix")

    test_from_dicts_saves_to_disk()
    print("✓ test_from_dicts_saves_to_disk")

    test_from_dicts_float_inference()
    print("✓ test_from_dicts_float_inference")

    test_from_dicts_nested_json_round_trip()
    print("✓ test_from_dicts_nested_json_round_trip")

    test_describe_table_with_sample_rows()
    print("✓ test_describe_table_with_sample_rows")

    test_list_tables_with_row_counts()
    print("✓ test_list_tables_with_row_counts")

    test_query_with_bound_parameters()
    print("✓ test_query_with_bound_parameters")

    test_query_max_rows_truncation()
    print("✓ test_query_max_rows_truncation")

    test_from_dicts_empty_rejects()
    print("✓ test_from_dicts_empty_rejects")

    test_tool_cache_same_tool_object()
    print("✓ test_tool_cache_same_tool_object")

    test_describe_nonexistent_table_returns_error()
    print("✓ test_describe_nonexistent_table_returns_error")

    test_missing_sql_for_query_returns_error()
    print("✓ test_missing_sql_for_query_returns_error")

    test_tsv_output_format()
    print("✓ test_tsv_output_format")

    test_from_dicts_writable_allows_insert()
    print("✓ test_from_dicts_writable_allows_insert")

    test_manager_raises_for_nonexistent_path()
    print("✓ test_manager_raises_for_nonexistent_path")

    print()
    print("=" * 60)
    print("SqliteManager live network tests (require API keys)")
    print("=" * 60)

    await test_live_agent_queries_from_dicts()
    print("✓ test_live_agent_queries_from_dicts")

    await test_live_agent_queries_db_on_disk()
    print("✓ test_live_agent_queries_db_on_disk")

    await test_live_agent_schema_discovery()
    print("✓ test_live_agent_schema_discovery")

    await test_live_agent_from_dicts_on_disk_path()
    print("✓ test_live_agent_from_dicts_on_disk_path")

    print()
    print("All SqliteManager tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
