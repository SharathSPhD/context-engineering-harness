import pytest
from src.avacchedaka.element import ContextElement, AvacchedakaConditions


def test_element_creation(sample_element):
    assert sample_element.id == "test-001"
    assert sample_element.precision == 0.9
    assert sample_element.sublated_by is None


def test_element_not_sublated_by_default(sample_element):
    assert sample_element.sublated_by is None


def test_element_avacchedaka_has_qualificand(sample_element):
    assert sample_element.avacchedaka.qualificand == "auth_module"


def test_element_precision_bounds():
    with pytest.raises(ValueError, match="precision must be between 0 and 1"):
        ContextElement(
            id="bad",
            content="x",
            precision=1.5,
            avacchedaka=AvacchedakaConditions(
                qualificand="x", qualifier="y", condition="z"
            ),
        )
