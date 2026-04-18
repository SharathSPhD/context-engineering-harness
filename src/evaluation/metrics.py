import numpy as np


def _get(example, key, default=None):
    if hasattr(example, key):
        return getattr(example, key)
    if isinstance(example, dict):
        return example.get(key, default)
    return default


def congruence_ratio(example) -> float:
    """Ratio of domain-congruent distractors to total passages (0 for incongruent examples)."""
    n_distractors = len(_get(example, "distractors", []))
    total = n_distractors + 1
    if _get(example, "version", "") == "congruent":
        return n_distractors / total if total > 0 else 0.0
    return 0.0


def expected_calibration_error(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    confs = np.array(confidences)
    correct = np.array(correctness, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Use <= for the last bin's upper edge so confidence=1.0 is captured
        upper = (confs <= bin_edges[i + 1]) if i == n_bins - 1 else (confs < bin_edges[i + 1])
        mask = (confs >= bin_edges[i]) & upper
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)
    return float(ece)
