"""Tests for memory subsystem migration and manager behavior."""

from __future__ import annotations

from pathlib import Path

from app.models.sql import MemoryType
from app.store.memory import MemoryManager, MemoryWriteRequest


def test_memory_migration_contains_required_tables() -> None:
    """Migration should define sessions and memory_items tables."""

    migration = Path("migrations/0002_memory_subsystem.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS sessions" in migration
    assert "CREATE TABLE IF NOT EXISTS memory_items" in migration
    assert "CREATE TYPE memory_type AS ENUM" in migration


def test_memory_manager_write_and_retrieve_orders_by_relevance() -> None:
    """Retrieve should return relevant memories ranked by overlap/importance."""

    manager = MemoryManager()
    manager.reset_fallback()
    manager.write(
        [
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="I prefer low-risk ETFs for long term growth",
                tags=["preference", "risk"],
                importance=0.9,
            ),
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="My cat name is Pixel",
                tags=["bio"],
                importance=0.2,
            ),
        ]
    )

    result = manager.retrieve(
        session_id="s1",
        tenant_id="t1",
        query_text="Can you build a low-risk ETF plan?",
        top_k=2,
    )

    assert len(result) == 2
    assert "low-risk ETFs" in result[0].content


def test_memory_manager_dedupes_exact_content_with_tag_overlap() -> None:
    """Exact content with overlapping tags should upsert rather than duplicate."""

    manager = MemoryManager()
    manager.reset_fallback()

    first = manager.write(
        [
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="I like weekly budget reviews",
                tags=["preference", "planning"],
                importance=0.6,
            )
        ]
    )
    second = manager.write(
        [
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="I like weekly budget reviews",
                tags=["preference", "habit"],
                importance=0.9,
            )
        ]
    )

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].id == second[0].id
    assert set(second[0].tags) == {"preference", "planning", "habit"}
    assert second[0].importance == 0.9


def test_memory_manager_summarize_session_updates_summary() -> None:
    """Summarize should persist a session summary string."""

    manager = MemoryManager()
    manager.reset_fallback()
    manager.write(
        [
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                content="User asked for 50/30/20 budget guidance",
                tags=["request"],
            ),
            MemoryWriteRequest(
                tenant_id="t1",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="User prefers concise answers",
                tags=["preference"],
            ),
        ]
    )

    summary = manager.summarize_session(session_id="s1", tenant_id="t1")

    assert "User asked for 50/30/20 budget guidance" in summary
    assert "User prefers concise answers" in summary
