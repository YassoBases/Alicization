"""Every module in the project must import cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "world",
    "world.config",
    "world.engine",
    "world.levers",
    "agent",
    "agent.encoder",
    "agent.core_gru",
    "agent.core_rssm",
    "agent.policy",
    "agent.drives",
    "ledger",
    "ledger.body_model",
    "ledger.attribution",
    "ledger.reliability",
    "ledger.forecaster",
    "memory",
    "memory.episodic",
    "training",
    "training.ppo",
    "training.replay",
    "training.reward",
    "training.sleep",
    "training.checkpoints",
    "training.loggers",
    "training.vecenv",
    "experiments",
    "experiments.runner",
    "experiments.batteries",
    "viz",
    "viz.viewer",
    "viz.dashboard",
    "viz.plots",
    "train",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
