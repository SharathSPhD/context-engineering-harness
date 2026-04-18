from unittest.mock import patch, MagicMock
import json


def _mock_cli(text: str):
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps({"result": text})
    m.stderr = ""
    return m


def test_h1_run_returns_required_keys():
    with patch("subprocess.run", return_value=_mock_cli("The token is valid for 24 hours.")):
        from experiments.validate.h1_schema import run_h1
        result = run_h1()
    assert "hypothesis" in result
    assert result["hypothesis"] == "H1"
    assert "congruent_accuracy" in result
    assert "incongruent_accuracy" in result
    assert "target_met" in result


def test_h2_run_returns_required_keys():
    with patch("subprocess.run", return_value=_mock_cli("The answer is 24 hours.")):
        from experiments.validate.h2_rag import run_h2
        result = run_h2()
    assert result["hypothesis"] == "H2"
    assert "precision_rag_accuracy" in result
    assert "vanilla_rag_accuracy" in result
    assert "target_met" in result


def test_h1_algorithmic_congruent_is_harder():
    from experiments.validate.h1_schema import _congruence_ratios
    ratios = _congruence_ratios()
    assert ratios["congruent"] > ratios["incongruent"]


def test_h2_algorithmic_precision_selects_correct():
    from experiments.validate.h2_rag import _algorithmic_selection_accuracy
    prec_acc, vanilla_acc = _algorithmic_selection_accuracy()
    assert prec_acc == 1.0
