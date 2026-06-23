"""
Tests for WritableMemory — CoALA-style writable semantic and episodic stores.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.core.memory.writable_memory import WritableMemory


# ======================================================================
# Unit tests: semantic memory
# ======================================================================


class TestSemanticMemory:
    def test_write_and_recall_rule(self) -> None:
        mem = WritableMemory()
        msg = mem.write_semantic_rule(
            rule_text="If SoC < 20% and in eclipse, switch to safe mode.",
            condition="battery_soc < 0.20 and not in_sunlight",
            action="safe",
        )
        assert "1" in msg  # confirmation includes count
        rules = mem.recall_semantic()
        assert len(rules) == 1
        assert rules[0]["rule"] == "If SoC < 20% and in eclipse, switch to safe mode."
        assert rules[0]["condition"] == "battery_soc < 0.20 and not in_sunlight"
        assert rules[0]["action"] == "safe"

    def test_multiple_rules_appended(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("Rule A")
        mem.write_semantic_rule("Rule B")
        mem.write_semantic_rule("Rule C")
        assert len(mem.recall_semantic()) == 3

    def test_recall_semantic_filter(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("battery rule: charge when low")
        mem.write_semantic_rule("observation rule: observe in sunlight")
        results = mem.recall_semantic(query="battery")
        assert len(results) == 1
        assert "battery" in results[0]["rule"]

    def test_recall_semantic_empty_query_returns_all(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("rule one")
        mem.write_semantic_rule("rule two")
        assert len(mem.recall_semantic(query="")) == 2

    def test_provenance_stored(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule(
            "some rule",
            provenance={"episode_id": 3, "step": 42},
        )
        rule = mem.recall_semantic()[0]
        assert rule["provenance"]["episode_id"] == 3

    def test_confirmation_message_increments(self) -> None:
        mem = WritableMemory()
        m1 = mem.write_semantic_rule("rule 1")
        m2 = mem.write_semantic_rule("rule 2")
        assert "1" in m1
        assert "2" in m2


# ======================================================================
# Unit tests: episodic memory
# ======================================================================


class TestEpisodicMemory:
    def test_write_and_recall_episode(self) -> None:
        mem = WritableMemory()
        msg = mem.write_episodic_entry(
            summary="Episode 0: mostly charging due to eclipse.",
            outcome="utility=0.45",
            episode_id=0,
        )
        assert "1" in msg
        entries = mem.recall_episodic()
        assert len(entries) == 1
        assert entries[0]["summary"] == "Episode 0: mostly charging due to eclipse."
        assert entries[0]["outcome"] == "utility=0.45"
        assert entries[0]["episode_id"] == 0

    def test_recall_episodic_most_recent_first(self) -> None:
        mem = WritableMemory()
        for i in range(5):
            mem.write_episodic_entry(summary=f"episode {i}")
        entries = mem.recall_episodic(last_n=3)
        assert len(entries) == 3
        assert entries[0]["summary"] == "episode 4"  # most recent first

    def test_episodic_ring_buffer_max_size(self) -> None:
        mem = WritableMemory(config={"episodic_max_size": 3})
        for i in range(5):
            mem.write_episodic_entry(summary=f"ep{i}")
        entries = mem.recall_episodic(last_n=10)
        assert len(entries) == 3  # only last 3 kept

    def test_recall_episodic_filter(self) -> None:
        mem = WritableMemory()
        mem.write_episodic_entry(summary="anomaly detected — safe mode triggered")
        mem.write_episodic_entry(summary="nominal pass — good downlink")
        results = mem.recall_episodic(query="anomaly")
        assert len(results) == 1
        assert "anomaly" in results[0]["summary"]

    def test_confirmation_message_increments(self) -> None:
        mem = WritableMemory()
        m1 = mem.write_episodic_entry("ep 1")
        m2 = mem.write_episodic_entry("ep 2")
        assert "1" in m1
        assert "2" in m2


# ======================================================================
# Unit tests: get_state and FixedMemory inheritance
# ======================================================================


class TestGetState:
    def test_get_state_includes_writable_stores(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("a rule")
        mem.write_episodic_entry("an episode")
        state = mem.get_state()
        assert "semantic_store" in state
        assert "episodic_store" in state
        assert len(state["semantic_store"]) == 1
        assert len(state["episodic_store"]) == 1

    def test_fixed_memory_slots_still_present(self) -> None:
        mem = WritableMemory()
        state = mem.get_state()
        assert "constellation_state" in state  # inherited FixedMemory slot


# ======================================================================
# Unit tests: reset does NOT clear writable stores
# ======================================================================


class TestReset:
    def test_reset_preserves_writable_stores(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("persistent rule")
        mem.write_episodic_entry("persistent episode")
        mem.reset()  # should clear working/task memory only
        assert len(mem.recall_semantic()) == 1
        assert len(mem.recall_episodic()) == 1

    def test_clear_learned_state_wipes_stores(self) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("rule")
        mem.write_episodic_entry("episode")
        mem.clear_learned_state()
        assert len(mem.recall_semantic()) == 0
        assert len(mem.recall_episodic()) == 0

    def test_reset_clears_working_memory(self) -> None:
        mem = WritableMemory()
        mem.update("constellation_state", {"sat": "ok"})
        mem.reset()
        assert mem.query("constellation_state") == {}


# ======================================================================
# Unit tests: persistence (save / load)
# ======================================================================


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("rule A", condition="x < 0.2", action="safe")
        mem.write_episodic_entry("ep 1", outcome="0.6", episode_id=1)

        path = str(tmp_path / "memory.json")
        mem.save(path)

        mem2 = WritableMemory()
        mem2.load(path)
        rules = mem2.recall_semantic()
        episodes = mem2.recall_episodic()

        assert len(rules) == 1
        assert rules[0]["rule"] == "rule A"
        assert len(episodes) == 1
        assert episodes[0]["episode_id"] == 1

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        mem = WritableMemory()
        mem.write_semantic_rule("rule")
        path = str(tmp_path / "subdir" / "memory.json")
        mem.save(path)
        assert Path(path).exists()

    def test_load_from_memory_path_config(self, tmp_path: Path) -> None:
        """WritableMemory auto-loads from memory_path if file exists."""
        path = str(tmp_path / "state.json")
        # Pre-populate file
        data = {
            "semantic_store": [{"rule": "preloaded rule", "condition": "", "action": "", "provenance": {}}],
            "episodic_store": [],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

        mem = WritableMemory(config={"memory_path": path})
        assert len(mem.recall_semantic()) == 1
        assert mem.recall_semantic()[0]["rule"] == "preloaded rule"

    def test_load_nonexistent_path_raises(self) -> None:
        mem = WritableMemory()
        with pytest.raises(FileNotFoundError):
            mem.load("/nonexistent/path/memory.json")


# ======================================================================
# Integration test: agentic loop with writable_coala writes to memory
# ======================================================================


class TestAgenticWritableCoalaIntegration:
    def test_memory_write_rule_tool_writes_to_writable_memory(self) -> None:
        """memory_write_rule tool appends a rule to WritableMemory."""
        from src.eventsat.agentic_tools import execute_tool
        from src.core.memory.writable_memory import WritableMemory

        mem = WritableMemory()
        state = {"battery_soc": 0.15, "in_sunlight": False}
        result = execute_tool(
            "memory_write_rule",
            {"rule_text": "Eclipse + low battery → safe", "condition": "soc<0.2", "action": "safe"},
            state,
            memory=mem,
        )
        assert result["status"] == "written"
        assert len(mem.recall_semantic()) == 1
        assert "Eclipse" in mem.recall_semantic()[0]["rule"]

    def test_memory_write_episode_tool_writes_to_writable_memory(self) -> None:
        """memory_write_episode tool appends an entry to WritableMemory."""
        from src.eventsat.agentic_tools import execute_tool
        from src.core.memory.writable_memory import WritableMemory

        mem = WritableMemory()
        state = {}
        result = execute_tool(
            "memory_write_episode",
            {"summary": "Nominal episode", "outcome": "utility=0.72"},
            state,
            memory=mem,
        )
        assert result["status"] == "written"
        entries = mem.recall_episodic()
        assert len(entries) == 1
        assert entries[0]["outcome"] == "utility=0.72"

    def test_memory_write_rule_without_writable_memory_returns_error(self) -> None:
        """Calling memory_write_rule with FixedMemory returns an error dict."""
        from src.eventsat.agentic_tools import execute_tool
        from src.core.memory.fixed_memory import FixedMemory

        mem = FixedMemory()
        result = execute_tool("memory_write_rule", {"rule_text": "test"}, {}, memory=mem)
        assert "error" in result

    def test_agentic_writable_coala_mock_uses_writable_memory(self) -> None:
        """AgenticEventSat with writable_coala mechanism creates WritableMemory."""
        from src.eventsat.agentic import AgenticEventSat
        from src.core.memory.writable_memory import WritableMemory

        rep = AgenticEventSat(config={
            "llm_mock": True,
            "behaviour_config": {"mechanism": "writable_coala"},
        })
        assert rep._mechanism == "writable_coala"
        assert isinstance(rep._memory, WritableMemory)

    def test_agentic_hand_designed_has_no_writable_memory(self) -> None:
        """Default hand_designed config has no internal WritableMemory."""
        from src.eventsat.agentic import AgenticEventSat

        rep = AgenticEventSat(config={"llm_mock": True})
        assert rep._mechanism == "hand_designed"
        assert rep._memory is None

    def test_agentic_prompt_optimized_falls_back_without_file(self) -> None:
        """prompt_optimized falls back to default prompt when file missing."""
        import warnings
        from src.eventsat.agentic import AgenticEventSat
        from src.eventsat.agentic_prompts import AGENTIC_SYSTEM_PROMPT

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = AgenticEventSat(config={
                "llm_mock": True,
                "experiment_id": "nonexistent_exp",
                "behaviour_config": {"mechanism": "prompt_optimized"},
            })
        assert rep._system_prompt == AGENTIC_SYSTEM_PROMPT
        assert any("trained prompt not found" in str(warning.message) for warning in w)

    def test_agentic_prompt_optimized_loads_prompt_from_file(self, tmp_path: Path) -> None:
        """prompt_optimized loads the optimised prompt from file."""
        from src.eventsat.agentic import AgenticEventSat

        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Custom optimised system prompt", encoding="utf-8")

        rep = AgenticEventSat(config={
            "llm_mock": True,
            "behaviour_config": {
                "mechanism": "prompt_optimized",
                "trained_prompt_path": str(prompt_path),
            },
        })
        assert rep._system_prompt == "Custom optimised system prompt"

    def test_writable_coala_system_prompt_includes_suffix(self) -> None:
        """writable_coala extends the system prompt with memory instructions."""
        from src.eventsat.agentic import AgenticEventSat

        rep = AgenticEventSat(config={
            "llm_mock": True,
            "behaviour_config": {"mechanism": "writable_coala"},
        })
        assert "memory_write_rule" in rep._system_prompt
        assert "memory_write_episode" in rep._system_prompt

    def test_writable_coala_tool_schemas_include_memory_tools(self) -> None:
        """writable_coala includes memory-write tools in the tool schema list."""
        from src.eventsat.agentic import AgenticEventSat

        rep = AgenticEventSat(config={
            "llm_mock": True,
            "behaviour_config": {"mechanism": "writable_coala"},
        })
        tool_names = {s["name"] for s in rep._tool_schemas}
        assert "memory_write_rule" in tool_names
        assert "memory_write_episode" in tool_names

    def test_hand_designed_tool_schemas_exclude_memory_tools(self) -> None:
        """Default hand_designed config does not expose memory-write tools."""
        from src.eventsat.agentic import AgenticEventSat

        rep = AgenticEventSat(config={"llm_mock": True})
        tool_names = {s["name"] for s in rep._tool_schemas}
        assert "memory_write_rule" not in tool_names
        assert "memory_write_episode" not in tool_names
