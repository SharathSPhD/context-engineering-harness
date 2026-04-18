from unittest.mock import patch, MagicMock
import json


def _mock_cli(text: str):
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps({"result": text})
    m.stderr = ""
    return m


def test_h3_returns_required_keys():
    two_stage_resp = json.dumps({"answer": "1 hour", "confidence": 0.9,
                                  "reasoning_trace": "ok", "sublated_candidates": [],
                                  "khyativada_flags": []})
    single_resp = json.dumps({"answer": "24 hours", "confidence": 0.7,
                               "reasoning_trace": "ok", "sublated_candidates": [],
                               "khyativada_flags": []})
    with patch("subprocess.run", side_effect=[
        _mock_cli('{"candidate_summary":"ok","uncertainty":0.3,'
                  '"recommended_queries":[],"reasoning_sketch":"jwt"}'),
        _mock_cli(two_stage_resp),
        _mock_cli('{"candidate_summary":"ok","uncertainty":0.3,'
                  '"recommended_queries":[],"reasoning_sketch":"jwt"}'),
        _mock_cli(single_resp),
        _mock_cli('{"candidate_summary":"ok","uncertainty":0.9,'
                  '"recommended_queries":[],"reasoning_sketch":"?"}'),
        _mock_cli(json.dumps({"answer": None, "confidence": 0.3,
                               "reasoning_trace": "no grounding",
                               "sublated_candidates": [], "khyativada_flags": []})),
        _mock_cli(json.dumps({"answer": "I don't know", "confidence": 0.4,
                               "reasoning_trace": "no grounding",
                               "sublated_candidates": [], "khyativada_flags": []})),
    ]):
        from experiments.validate.h3_agents import run_h3
        result = run_h3()
    assert result["hypothesis"] == "H3"
    assert "two_stage_accuracy" in result
    assert "single_stage_accuracy" in result
    assert "target_met" in result


def test_h4_returns_required_keys():
    from experiments.validate.h4_compaction import run_h4
    result = run_h4()  # no CLI calls needed
    assert result["hypothesis"] == "H4"
    assert "boundary_retention" in result
    assert "threshold_retention" in result
    assert "target_met" in result


def test_h4_boundary_retains_more_post_boundary():
    from experiments.validate.h4_compaction import _measure_retention
    boundary_ret, threshold_ret = _measure_retention()
    assert boundary_ret >= threshold_ret
