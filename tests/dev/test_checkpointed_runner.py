"""Unit tests for tools.dev.runners.checkpointed_runner.

Verifies per-example JSONL append, resume-from-checkpoint, and graceful
QuotaExhausted handling that writes a ``*_partial.json`` receipt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkExample,
    ModelOutput,
)
from src.benchmarks.hypothesis import HypothesisSpec, TargetDirection
from src.benchmarks.runner import RunnerConfig
from tools.dev.runners import (
    CheckpointRecord,
    CheckpointedBundleRunner,
    PartialRunExit,
    load_checkpoint,
)
from tools.dev.scheduler import QuotaExhausted


# ---------------------------------------------------------------- fixture adapter


@dataclass
class _ToyAdapter(BenchmarkAdapter):
    """Two-example adapter with a transparent scoring function."""

    name: str = "toy_adapter"

    def load_examples(
        self, *, n: int | None = None, seed: int = 0
    ) -> list[BenchmarkExample]:
        base = [
            BenchmarkExample(id=f"s{seed}-ex{i}", prompt=f"q{i}", ground_truth=f"g{i}")
            for i in range(n or 2)
        ]
        return base

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        return f"[{condition}] {example.prompt}"

    def system_prompt(self, *, condition: str) -> str:
        return f"sys-{condition}"

    def score(self, example, output):
        ok = example.ground_truth in output.text
        return (1.0 if ok else 0.0), ok, output.text


# ---------------------------------------------------------------- helpers


def _spec(*, models=("mA",), seeds=(0,), n=2) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="H_test",
        description="checkpointed-runner unit test",
        adapter_name="toy_adapter",
        treatment_condition="harness_on",
        baseline_condition="harness_off",
        metric="accuracy",
        direction=TargetDirection.GREATER,
        delta=0.0,
        n_examples=n,
        seeds=seeds,
        models=models,
    )


def _oracle_caller(
    *, prompt, model, max_tokens, system="", seed=None
) -> ModelOutput:
    # Reply with every GT we see in the prompt.
    # The fixture prompt is literally "[condition] qN"; we echo back gN.
    idx = prompt.rsplit("q", 1)[-1]
    return ModelOutput(text=f"g{idx}", input_tokens=1, output_tokens=1)


def _always_fail_caller(
    *, prompt, model, max_tokens, system="", seed=None
) -> ModelOutput:
    return ModelOutput(text="WRONG", input_tokens=1, output_tokens=1)


# ---------------------------------------------------------------- tests


def test_checkpoint_appended_per_example(tmp_path):
    checkpoint = tmp_path / "_checkpoint_live.jsonl"
    runner = CheckpointedBundleRunner(
        label="H_test_toy",
        adapter=_ToyAdapter(),
        caller=_oracle_caller,
        spec=_spec(models=("mA",), seeds=(0,), n=2),
        checkpoint_path=checkpoint,
        runner_cfg=RunnerConfig(
            max_tokens=8, bootstrap_n=50, permutation_n=50
        ),
        resume=False,
    )
    payload = runner.run_bundle()

    assert payload["status"] == "complete"
    assert payload["outcome"] is not None
    assert checkpoint.exists()
    lines = [
        json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()
    ]
    # 1 model × 1 seed × 2 conditions × 2 examples = 4 rows.
    assert len(lines) == 4
    assert {row["example_id"] for row in lines} == {"s0-ex0", "s0-ex1"}
    assert {row["condition"] for row in lines} == {"harness_on", "harness_off"}


def test_resume_skips_completed_keys_and_does_not_recall(tmp_path):
    checkpoint = tmp_path / "_checkpoint_live.jsonl"
    # Seed the checkpoint with a fake row for (mA, 0, harness_on, s0-ex0).
    prelim = CheckpointRecord(
        adapter="toy_adapter",
        model="mA",
        seed=0,
        condition="harness_on",
        example_id="s0-ex0",
        score=1.0,
        correct=True,
        prediction="g0",
        input_tokens=7,
        output_tokens=3,
        latency_ms=42.0,
        ts="2026-04-18T00:00:00+00:00",
        git_sha="deadbeef",
    )
    checkpoint.write_text(json.dumps(prelim.__dict__) + "\n")

    calls: list[str] = []

    def recording_caller(
        *, prompt, model, max_tokens, system="", seed=None
    ) -> ModelOutput:
        calls.append(prompt)
        idx = prompt.rsplit("q", 1)[-1]
        return ModelOutput(text=f"g{idx}", input_tokens=2, output_tokens=1)

    runner = CheckpointedBundleRunner(
        label="H_test_toy",
        adapter=_ToyAdapter(),
        caller=recording_caller,
        spec=_spec(models=("mA",), seeds=(0,), n=2),
        checkpoint_path=checkpoint,
        runner_cfg=RunnerConfig(max_tokens=8, bootstrap_n=50, permutation_n=50),
        resume=True,
    )
    payload = runner.run_bundle()

    # 4 total keys; one was pre-seeded, so only 3 real calls are made.
    assert len(calls) == 3
    assert payload["status"] == "complete"
    # The pre-seeded row's tokens must be preserved in the rolled-up run.
    treatment_rollups = [
        r for r in payload["treatment_runs"] if r["model"] == "mA" and r["seed"] == 0
    ]
    assert treatment_rollups and treatment_rollups[0]["total_input_tokens"] >= 7


def test_quota_exhausted_mid_bundle_writes_partial_and_raises(tmp_path):
    checkpoint = tmp_path / "_checkpoint_live.jsonl"
    n_calls = {"count": 0}

    def fail_second_caller(
        *, prompt, model, max_tokens, system="", seed=None
    ) -> ModelOutput:
        n_calls["count"] += 1
        if n_calls["count"] >= 3:
            raise QuotaExhausted(
                "window exhausted", reason="simulated", window_summary={"n_calls": 2}
            )
        idx = prompt.rsplit("q", 1)[-1]
        return ModelOutput(text=f"g{idx}", input_tokens=1, output_tokens=1)

    runner = CheckpointedBundleRunner(
        label="H_test_toy",
        adapter=_ToyAdapter(),
        caller=fail_second_caller,
        spec=_spec(models=("mA",), seeds=(0,), n=2),
        checkpoint_path=checkpoint,
        runner_cfg=RunnerConfig(max_tokens=8, bootstrap_n=50, permutation_n=50),
        resume=False,
    )

    with pytest.raises(PartialRunExit) as exc_info:
        runner.run_bundle()

    assert exc_info.value.reason == "simulated"
    assert exc_info.value.partial_path.exists()

    saved = json.loads(exc_info.value.partial_path.read_text())
    assert saved["status"] == "partial_quota"
    assert saved["partial"]["reason"] == "simulated"
    # 2 calls succeeded before quota hit -> 2 checkpoint rows.
    rows = [
        json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()
    ]
    assert len(rows) == 2


def test_load_checkpoint_tolerates_malformed_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    good = CheckpointRecord(
        adapter="x",
        model="m",
        seed=0,
        condition="harness_on",
        example_id="e0",
        score=1.0,
        correct=True,
        prediction="y",
        input_tokens=1,
        output_tokens=1,
        latency_ms=0.0,
        ts="",
        git_sha="",
    )
    path.write_text(
        "\n".join(
            [
                "{not json",
                "{}",  # missing required keys
                json.dumps(good.__dict__),
                "",
            ]
        )
    )
    loaded = load_checkpoint(path)
    assert len(loaded) == 1
    assert good.key() in loaded
