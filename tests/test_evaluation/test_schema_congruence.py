import pytest
from src.evaluation.schema_congruence import CongruenceBenchmarkBuilder, BenchmarkExample
from src.evaluation.metrics import congruence_ratio, expected_calibration_error


def test_congruent_example_uses_domain_distractors():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example("JWT tokens expire after 24h.", "web_security", 4, "congruent")
    assert example.version == "congruent"
    assert example.gold_passage in example.context
    assert len(example.distractors) > 0


def test_incongruent_example_uses_unrelated_distractors():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example("JWT tokens expire after 24h.", "web_security", 4, "incongruent")
    assert example.version == "incongruent"
    assert example.gold_passage in example.context


def test_gold_passage_always_in_context():
    builder = CongruenceBenchmarkBuilder()
    for version in ("congruent", "incongruent"):
        example = builder.build_example("Test gold.", "web_security", 4, version)
        assert "Test gold." in example.context


def test_invalid_version_raises():
    builder = CongruenceBenchmarkBuilder()
    with pytest.raises(ValueError, match="version must be"):
        builder.build_example("x", "web_security", 4, "random")


def test_congruence_ratio_congruent():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example("JWT.", "web_security", 4, "congruent")
    ratio = congruence_ratio(example)
    assert 0.0 < ratio < 1.0


def test_congruence_ratio_incongruent_is_zero():
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example("JWT.", "web_security", 4, "incongruent")
    assert congruence_ratio(example) == 0.0


def test_expected_calibration_error_perfect():
    # Perfect calibration: confidence = accuracy at every point
    ece = expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, True, False, False])
    assert ece < 0.05


def test_expected_calibration_error_overconfident():
    # Overconfident: high confidence but wrong
    ece = expected_calibration_error([0.9, 0.9], [False, False])
    assert ece > 0.5


def test_reproducible_seed():
    b1 = CongruenceBenchmarkBuilder(seed=42)
    b2 = CongruenceBenchmarkBuilder(seed=42)
    e1 = b1.build_example("x", "web_security", 4, "congruent")
    e2 = b2.build_example("x", "web_security", 4, "congruent")
    assert e1.distractors == e2.distractors


def test_benchmark_example_supports_subscript_access():
    """BenchmarkExample must support dict-style subscript access per API contract."""
    builder = CongruenceBenchmarkBuilder()
    example = builder.build_example("JWT tokens expire after 24h.", "web_security", 4, "congruent")
    assert example["version"] == "congruent"
    assert example["gold_passage"] == "JWT tokens expire after 24h."
    assert isinstance(example["distractors"], list)
