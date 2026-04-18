import pytest
from src.compaction.detector import EventBoundaryDetector


def test_high_surprise_triggers_boundary():
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    surprises = [0.2, 0.3, 0.25, 0.9, 0.2]
    boundaries = detector.detect_from_surprises(surprises)
    assert 3 in boundaries


def test_no_boundary_when_surprises_low():
    detector = EventBoundaryDetector(surprise_threshold=0.8)
    surprises = [0.2, 0.3, 0.25, 0.4, 0.2]
    boundaries = detector.detect_from_surprises(surprises)
    assert len(boundaries) == 0


def test_task_switch_triggers_boundary():
    detector = EventBoundaryDetector()
    assert detector.detect_from_signals(task_switch=True, surprise_spike=False) is True


def test_surprise_spike_triggers_boundary():
    detector = EventBoundaryDetector()
    assert detector.detect_from_signals(task_switch=False, surprise_spike=True) is True


def test_neither_signal_no_boundary():
    detector = EventBoundaryDetector()
    assert detector.detect_from_signals(task_switch=False, surprise_spike=False) is False


def test_threshold_at_exact_value_triggers():
    detector = EventBoundaryDetector(surprise_threshold=0.75)
    surprises = [0.75]
    boundaries = detector.detect_from_surprises(surprises)
    assert 0 in boundaries


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="surprise_threshold"):
        EventBoundaryDetector(surprise_threshold=0.0)


def test_window_average_smooths():
    detector = EventBoundaryDetector()
    surprises = [0.0, 0.0, 1.0, 0.0, 0.0]
    averaged = detector.window_average_surprise(surprises, window=4)
    assert len(averaged) == len(surprises)
    assert max(averaged) < 1.0


def test_window_average_empty_list():
    detector = EventBoundaryDetector()
    assert detector.window_average_surprise([]) == []
