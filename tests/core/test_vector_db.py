"""Tests for the in-process vector DB and VectorDBManager tool wiring."""

from __future__ import annotations

import json
import sys
import uuid

import numpy as np

sys.path.insert(0, "src")

from lm_deluge.tool.prefab.vector_db import (
    InProcessVectorDB,
    VectorDBManager,
    VectorDBRecord,
)


def _make_vec(dim: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).tolist()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


# ---------------------------------------------------------------------------
# InProcessVectorDB unit tests
# ---------------------------------------------------------------------------


def test_insert_and_count():
    db = InProcessVectorDB(dimension=4)
    assert db.count() == 0

    ids = db.insert(
        [
            VectorDBRecord(id="a", text="hello", vector=[1, 0, 0, 0]),
            VectorDBRecord(id="b", text="world", vector=[0, 1, 0, 0]),
        ]
    )
    assert ids == ["a", "b"]
    assert db.count() == 2


def test_auto_dimension():
    db = InProcessVectorDB()
    assert db.dimension is None
    db.insert([VectorDBRecord(id="x", text="t", vector=[1, 2, 3])])
    assert db.dimension == 3


def test_dimension_mismatch():
    db = InProcessVectorDB(dimension=3)
    try:
        db.insert([VectorDBRecord(id="x", text="t", vector=[1, 2])])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e).lower()


def test_duplicate_id():
    db = InProcessVectorDB(dimension=2)
    db.insert([VectorDBRecord(id="a", text="t", vector=[1, 0])])
    try:
        db.insert([VectorDBRecord(id="a", text="t2", vector=[0, 1])])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Duplicate" in str(e)


def test_search_cosine():
    db = InProcessVectorDB(dimension=3)
    db.insert(
        [
            VectorDBRecord(id="a", text="north", vector=[1, 0, 0]),
            VectorDBRecord(id="b", text="east", vector=[0, 1, 0]),
            VectorDBRecord(id="c", text="up", vector=[0, 0, 1]),
        ]
    )

    results = db.search([1, 0.1, 0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "a"  # most similar to [1, 0, 0]
    assert results[0].score > results[1].score


def test_search_empty_db():
    db = InProcessVectorDB(dimension=3)
    results = db.search([1, 0, 0], top_k=5)
    assert results == []


def test_get():
    db = InProcessVectorDB(dimension=2)
    db.insert(
        [
            VectorDBRecord(id="a", text="hello", vector=[1, 0]),
            VectorDBRecord(id="b", text="world", vector=[0, 1]),
        ]
    )

    records = db.get(["b", "missing", "a"])
    assert records[0] is not None and records[0].id == "b"
    assert records[1] is None
    assert records[2] is not None and records[2].id == "a"


def test_delete():
    db = InProcessVectorDB(dimension=2)
    db.insert(
        [
            VectorDBRecord(id="a", text="hello", vector=[1, 0]),
            VectorDBRecord(id="b", text="world", vector=[0, 1]),
            VectorDBRecord(id="c", text="foo", vector=[1, 1]),
        ]
    )
    assert db.count() == 3

    removed = db.delete(["b", "nonexistent"])
    assert removed == 1
    assert db.count() == 2

    # b should be gone
    assert db.get(["b"]) == [None]

    # search should still work
    results = db.search([1, 0], top_k=5)
    assert len(results) == 2
    result_ids = {r.id for r in results}
    assert "a" in result_ids
    assert "c" in result_ids
    assert "b" not in result_ids


def test_list_ids():
    db = InProcessVectorDB(dimension=2)
    db.insert(
        [
            VectorDBRecord(id="a", text="t", vector=[1, 0]),
            VectorDBRecord(id="b", text="t", vector=[0, 1]),
            VectorDBRecord(id="c", text="t", vector=[1, 1]),
        ]
    )
    assert db.list_ids() == ["a", "b", "c"]
    assert db.list_ids(limit=2) == ["a", "b"]
    assert db.list_ids(limit=2, offset=1) == ["b", "c"]


def test_metadata():
    db = InProcessVectorDB(dimension=2)
    db.insert(
        [
            VectorDBRecord(
                id="a",
                text="hello",
                vector=[1, 0],
                metadata={"source": "test", "page": 1},
            ),
        ]
    )
    rec = db.get(["a"])[0]
    assert rec is not None
    assert rec.metadata == {"source": "test", "page": 1}

    results = db.search([1, 0], top_k=1)
    assert results[0].metadata == {"source": "test", "page": 1}


# ---------------------------------------------------------------------------
# VectorDBManager / tool wiring tests
# ---------------------------------------------------------------------------


def test_manager_get_tools():
    db = InProcessVectorDB(dimension=3)
    manager = VectorDBManager(db)
    tools = manager.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "vector_db"


def test_manager_exclude_commands():
    db = InProcessVectorDB(dimension=3)
    manager = VectorDBManager(db)
    tools = manager.get_tools(exclude=["delete"])
    assert len(tools) == 1
    # Using a delete should fail
    result = json.loads(tools[0].run(command="delete", ids=["x"]))
    assert result["ok"] is False
    assert "disabled" in result["message"]


def test_manager_insert_and_search():
    db = InProcessVectorDB(dimension=3)
    manager = VectorDBManager(db)
    tool = manager.get_tools()[0]

    # Insert
    result = json.loads(
        tool.run(
            command="insert",
            entries=[
                {"id": "a", "text": "north", "vector": [1, 0, 0]},
                {"id": "b", "text": "east", "vector": [0, 1, 0]},
            ],
        )
    )
    assert result["ok"] is True
    assert result["result"]["inserted_count"] == 2

    # Search
    result = json.loads(
        tool.run(
            command="search",
            query_vector=[1, 0.1, 0],
            top_k=1,
        )
    )
    assert result["ok"] is True
    assert result["result"]["results"][0]["id"] == "a"


def test_manager_get_delete_count():
    db = InProcessVectorDB(dimension=2)
    manager = VectorDBManager(db)
    tool = manager.get_tools()[0]

    tool.run(
        command="insert",
        entries=[
            {"id": "x", "text": "t", "vector": [1, 0]},
            {"id": "y", "text": "t", "vector": [0, 1]},
        ],
    )

    # Get
    result = json.loads(tool.run(command="get", ids=["x", "missing"]))
    assert result["ok"] is True
    assert result["result"]["records"][0]["id"] == "x"
    assert result["result"]["records"][1] is None

    # Count
    result = json.loads(tool.run(command="count"))
    assert result["ok"] is True
    assert result["result"]["total"] == 2

    # Delete
    result = json.loads(tool.run(command="delete", ids=["x"]))
    assert result["ok"] is True
    assert result["result"]["deleted_count"] == 1

    # Count after delete
    result = json.loads(tool.run(command="count"))
    assert result["result"]["total"] == 1


def test_manager_list_ids():
    db = InProcessVectorDB(dimension=2)
    manager = VectorDBManager(db)
    tool = manager.get_tools()[0]

    tool.run(
        command="insert",
        entries=[
            {"id": "a", "text": "t", "vector": [1, 0]},
            {"id": "b", "text": "t", "vector": [0, 1]},
            {"id": "c", "text": "t", "vector": [1, 1]},
        ],
    )

    result = json.loads(tool.run(command="list_ids", limit=2, offset=1))
    assert result["ok"] is True
    assert result["result"]["ids"] == ["b", "c"]


def test_manager_auto_id():
    db = InProcessVectorDB(dimension=2)
    manager = VectorDBManager(db)
    tool = manager.get_tools()[0]

    result = json.loads(
        tool.run(
            command="insert",
            entries=[{"text": "no id given", "vector": [1, 0]}],
        )
    )
    assert result["ok"] is True
    generated_id = result["result"]["ids"][0]
    # Should be a valid UUID
    uuid.UUID(generated_id)


def test_manager_error_handling():
    db = InProcessVectorDB(dimension=2)
    manager = VectorDBManager(db)
    tool = manager.get_tools()[0]

    # Missing required field
    result = json.loads(tool.run(command="search"))
    assert result["ok"] is False

    # Missing entries
    result = json.loads(tool.run(command="insert"))
    assert result["ok"] is False


def test_duplicate_id_within_batch():
    """Duplicate IDs within the same insert batch should be rejected."""
    db = InProcessVectorDB(dimension=2)
    try:
        db.insert(
            [
                VectorDBRecord(id="dup", text="first", vector=[1, 0]),
                VectorDBRecord(id="dup", text="second", vector=[0, 1]),
            ]
        )
        assert False, "Should have raised ValueError for intra-batch duplicate"
    except ValueError as e:
        assert "Duplicate" in str(e)
    # DB should be unchanged
    assert db.count() == 0


def test_get_returns_original_vectors():
    """get() should return the original vectors, not normalized ones."""
    db = InProcessVectorDB(dimension=2)
    original = [3.0, 4.0]
    db.insert([VectorDBRecord(id="a", text="t", vector=original)])
    rec = db.get(["a"])[0]
    assert rec is not None
    assert rec.vector == original


def test_delete_duplicate_ids_correct_count():
    """delete() with the same ID repeated should not over-count."""
    db = InProcessVectorDB(dimension=2)
    db.insert(
        [
            VectorDBRecord(id="a", text="t", vector=[1, 0]),
            VectorDBRecord(id="b", text="t", vector=[0, 1]),
        ]
    )
    removed = db.delete(["a", "a", "a"])
    assert removed == 1
    assert db.count() == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"  {test.__name__} ...", end=" ", flush=True)
        test()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
