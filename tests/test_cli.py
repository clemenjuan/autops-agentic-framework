"""
Tests for the AUTOPS CLI — entry point, subcommand dispatch, train command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ======================================================================
# Helper: write a minimal experiment YAML to tmp_path
# ======================================================================


def _write_config(
    tmp_path: Path,
    name: str = "test_exp",
    extra: dict | None = None,
) -> Path:
    data = {
        "experiment_id": name,
        "agent_organization": "sas",
        "decision_loop": "sda",
        "representation": "symbolic",
        "emergence_mode": "hand_designed",
        "operations_paradigm": "autonomous_hybrid",
        "num_episodes": 1,
        "max_steps": 2,
        "output_dir": str(tmp_path / "results"),
    }
    if extra:
        data.update(extra)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ======================================================================
# Basic CLI structure
# ======================================================================


class TestCLIStructure:
    def test_main_help_exits_zero(self, capsys) -> None:
        from src.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["autops", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        from src.cli import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["autops", "nonexistent"]):
                main()
        assert exc_info.value.code != 0

    def test_train_subcommand_registered(self) -> None:
        """Verify `train` appears as a registered subcommand."""
        from src.cli import main
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["autops", "train", "--help"]):
                main()


# ======================================================================
# cmd_run smoke test
# ======================================================================


class TestCmdRun:
    def test_run_minimal_config(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, "run_test")
        from src.cli import cmd_run
        args = MagicMock()
        args.config = str(config_path)
        args.episodes = 1
        args.steps = 2
        args.seed = None
        args.output_dir = str(tmp_path / "results")
        args.log_level = None
        args.analyze = False
        cmd_run(args)  # should not raise
        assert (tmp_path / "results" / "results.json").exists()


# ======================================================================
# cmd_train: PPO dispatch
# ======================================================================


class TestCmdTrainPPO:
    def test_train_ppo_calls_ppo_trainer(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            "subm_le_test",
            extra={
                "representation": "subsymbolic",
                "emergence_mode": "learned",
                "representation_config": {"type": "subsymbolic_eventsat"},
                "emergence_config": {"mechanism": "ppo"},
            },
        )
        from src.cli import cmd_train

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = "data/trained_models/subm_le_test/policy.pt"

        with patch("src.behaviour.training_pipeline.PPOTrainer", return_value=mock_trainer):
            args = MagicMock()
            args.config = str(config_path)
            args.timesteps = 1000
            args.source_dir = None
            args.seed = None
            args.output_dir = None
            cmd_train(args)

        mock_trainer.train.assert_called_once()

    def test_train_ppo_missing_torch_exits(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            "subm_le_notorch",
            extra={
                "representation": "subsymbolic",
                "emergence_mode": "learned",
                "representation_config": {"type": "subsymbolic_eventsat"},
                "emergence_config": {"mechanism": "ppo"},
            },
        )
        from src.cli import cmd_train

        # Make the import of training_pipeline fail (simulates missing torch)
        with patch.dict("sys.modules", {"src.behaviour.training_pipeline": None}):
            args = MagicMock()
            args.config = str(config_path)
            args.timesteps = None
            args.source_dir = None
            args.seed = None
            args.output_dir = None
            with pytest.raises(SystemExit) as exc_info:
                cmd_train(args)
            assert exc_info.value.code == 1


# ======================================================================
# cmd_train: prompt_optimized dispatch
# ======================================================================


class TestCmdTrainPromptOptimized:
    def test_train_prompt_optimized_calls_optimizer(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            "hybr_lep_test",
            extra={
                "representation": "hybrid",
                "emergence_mode": "learned",
                "representation_config": {"type": "llm_eventsat"},
                "emergence_config": {"mechanism": "prompt_optimized"},
            },
        )
        from src.cli import cmd_train

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = "Optimised prompt text"

        with patch("src.behaviour.prompt_optimizer.PromptOptimizer", return_value=mock_optimizer):
            args = MagicMock()
            args.config = str(config_path)
            args.timesteps = None
            args.source_dir = str(tmp_path / "source")
            args.seed = None
            args.output_dir = None
            cmd_train(args)

        mock_optimizer.optimize.assert_called_once()

    def test_train_prompt_optimized_no_source_dir_ok(self, tmp_path: Path) -> None:
        """Auto-derives source dir from experiment_id — should not raise."""
        config_path = _write_config(
            tmp_path,
            "agnt_lep_test",
            extra={
                "representation": "hybrid",
                "emergence_mode": "learned",
                "representation_config": {"type": "agentic_eventsat"},
                "emergence_config": {"mechanism": "prompt_optimized"},
            },
        )
        from src.cli import cmd_train

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = "Prompt"

        with patch("src.behaviour.prompt_optimizer.PromptOptimizer", return_value=mock_optimizer):
            args = MagicMock()
            args.config = str(config_path)
            args.timesteps = None
            args.source_dir = None
            args.seed = None
            args.output_dir = None
            cmd_train(args)  # should not raise

        mock_optimizer.optimize.assert_called_once()


# ======================================================================
# cmd_train: writable_coala dispatch
# ======================================================================


class TestCmdTrainWritableCoala:
    def test_writable_coala_no_training_prints_message(
        self, tmp_path: Path, capsys
    ) -> None:
        config_path = _write_config(
            tmp_path,
            "agnt_lec_test",
            extra={
                "representation": "hybrid",
                "emergence_mode": "learned",
                "representation_config": {"type": "agentic_eventsat"},
                "emergence_config": {"mechanism": "writable_coala"},
            },
        )
        from src.cli import cmd_train

        args = MagicMock()
        args.config = str(config_path)
        args.timesteps = None
        args.source_dir = None
        args.seed = None
        args.output_dir = None
        cmd_train(args)  # should not raise or exit

        captured = capsys.readouterr()
        assert "writable_coala" in captured.out
        assert "online" in captured.out.lower()


# ======================================================================
# cmd_train: unknown combination
# ======================================================================


class TestCmdTrainUnknown:
    def test_unknown_combination_exits(self, tmp_path: Path) -> None:
        config_path = _write_config(
            tmp_path,
            "weird_test",
            extra={
                "representation": "symbolic",
                "emergence_mode": "hand_designed",
                "representation_config": {"type": "rule_based_eventsat"},
                "emergence_config": {},
            },
        )
        from src.cli import cmd_train

        args = MagicMock()
        args.config = str(config_path)
        args.timesteps = None
        args.source_dir = None
        args.seed = None
        args.output_dir = None
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 1
