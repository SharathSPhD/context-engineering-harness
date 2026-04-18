"""Tests for src/config.py config loader."""
import importlib
from pathlib import Path
from unittest.mock import patch


def _reload_config(toml_text: str | None = None):
    """Helper: reload src.config with optional TOML override."""
    import sys
    # Remove cached module
    for key in list(sys.modules.keys()):
        if key.startswith("src.config"):
            del sys.modules[key]

    if toml_text is not None:
        mock_path = Path("/tmp/_test_config.toml")
        mock_path.write_text(toml_text)
        with patch("src.config._CONFIG_PATH", mock_path):
            import src.config as cfg
            importlib.reload(cfg)
            return cfg
    else:
        # No config file — use defaults
        with patch("src.config._CONFIG_PATH", Path("/nonexistent/config.toml")):
            import src.config as cfg
            importlib.reload(cfg)
            return cfg


def test_defaults_when_no_config_file():
    cfg = _reload_config(toml_text=None)
    assert cfg.config.fast_model == "claude-haiku-4-5"
    assert cfg.config.smart_model == "claude-sonnet-4-6"
    assert cfg.config.compress_threshold == 0.3
    assert cfg.config.surprise_threshold == 0.75
    assert cfg.config.random_seed == 42


def test_user_can_override_fast_model():
    cfg = _reload_config('[models]\nfast = "claude-opus-4-5"\n')
    assert cfg.config.fast_model == "claude-opus-4-5"
    assert cfg.config.smart_model == "claude-sonnet-4-6"  # still default


def test_user_can_override_smart_model():
    cfg = _reload_config('[models]\nsmart = "claude-opus-4-5"\n')
    assert cfg.config.smart_model == "claude-opus-4-5"


def test_user_can_override_thresholds():
    cfg = _reload_config('[compaction]\nsurprise_threshold = 0.9\n')
    assert cfg.config.surprise_threshold == 0.9
    assert cfg.config.compress_threshold == 0.3  # unchanged


def test_config_fast_model_used_as_manas_default():
    """ManasAgent default model must come from config, not hardcode."""
    import inspect
    from src.agents.manas import ManasAgent
    sig = inspect.signature(ManasAgent.__init__)
    # The default must be a reference to config.fast_model, not a literal string
    # We verify by checking that ManasAgent() uses whatever config says
    # (we can't easily check the default value itself without reloading)
    agent = ManasAgent()
    from src.config import config
    assert agent.model == config.fast_model


def test_config_smart_model_used_as_buddhi_default():
    from src.agents.buddhi import BuddhiAgent
    agent = BuddhiAgent()
    from src.config import config
    assert agent.model == config.smart_model
