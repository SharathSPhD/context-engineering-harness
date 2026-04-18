def test_h5_returns_required_keys():
    from experiments.validate.h5_multiagent import run_h5
    result = run_h5()
    assert result["hypothesis"] == "H5"
    assert "with_avacchedaka_conflict_rate" in result
    assert "without_avacchedaka_conflict_rate" in result
    assert "reduction_pct" in result
    assert result["target_met"] is True  # 100% reduction ≥ 30%


def test_h5_conflict_rate_without_is_100_pct():
    from experiments.validate.h5_multiagent import run_h5
    result = run_h5()
    assert result["without_avacchedaka_conflict_rate"] == 1.0


def test_h5_conflict_rate_with_is_0_pct():
    from experiments.validate.h5_multiagent import run_h5
    result = run_h5()
    assert result["with_avacchedaka_conflict_rate"] == 0.0


def test_h6_returns_required_keys():
    from experiments.validate.h6_classifier import run_h6
    result = run_h6()
    assert result["hypothesis"] == "H6"
    assert "accuracy" in result
    assert "n_correct" in result
    assert "n_total" in result


def test_h6_accuracy_above_threshold():
    from experiments.validate.h6_classifier import run_h6
    result = run_h6()
    assert result["accuracy"] >= 0.5


def test_h7_returns_required_keys():
    from experiments.validate.h7_forgetting import run_h7
    result = run_h7()
    assert result["hypothesis"] == "H7"
    assert "badha_first_retention" in result
    assert "no_forgetting_retention" in result
    assert "target_met" in result
