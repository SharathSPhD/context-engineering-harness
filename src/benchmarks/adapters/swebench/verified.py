"""SWE-bench Verified adapter — agent-style code-patch generation.

SWE-bench Verified (Carlini et al., 2024 — Princeton+OpenAI) is the
500-instance human-curated subset of SWE-bench used by every public
coding-agent leaderboard. Each instance is a real GitHub issue plus the
maintainer's gold patch and the test set the patch must make pass
(FAIL_TO_PASS) without breaking the test set it must not regress
(PASS_TO_PASS).

The adapter exposes two scoring paths:

  * `score(...)` — a fast, offline heuristic. Compares the model's
    proposed unified diff to the gold patch on (a) target file overlap
    and (b) line-level Jaccard similarity of the changed regions. This
    is what runs in CI and in the harness-on/harness-off A/B exploration
    runs that don't have docker. It is NOT what we publish.
  * `verify_with_swebench_harness(...)` — the official SWE-bench docker
    harness wrapper used for the published P6-C numbers. Raises
    SwebenchHarnessUnavailable when docker or the harness package is
    not present, so callers can degrade explicitly instead of silently
    reporting heuristic numbers as if they were validated.

Conditions:
  - "harness_on": prompt frames the task as "investigate, then patch",
    instructs the agent to (1) read existing code, (2) write a minimal
    diff, (3) cite the file path explicitly. This is the harness's
    plan-then-act discipline.
  - "harness_off": naive "here's an issue, write a patch" prompt.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.benchmarks.base import BenchmarkAdapter, BenchmarkExample, ModelOutput
from src.benchmarks.registry import register

from ..longctx._hf_loader import HFUnavailable, load_hf_examples
from ._synthetic import generate_swe_examples

logger = logging.getLogger(__name__)


class SwebenchHarnessUnavailable(RuntimeError):
    """Raised when the docker-backed SWE-bench harness cannot run."""


_DIFF_FILE_RE = re.compile(r"^(?:diff --git a/|---\s+a/|\+\+\+\s+b/)(\S+)", re.MULTILINE)
_DIFF_HUNK_LINE_RE = re.compile(r"^[+-](?![+-]{2})(.+)$", re.MULTILINE)


def _files_in_diff(diff: str) -> set[str]:
    return {m.group(1) for m in _DIFF_FILE_RE.finditer(diff)}


def _changed_lines(diff: str) -> list[str]:
    return [m.group(1).strip() for m in _DIFF_HUNK_LINE_RE.finditer(diff) if m.group(1).strip()]


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@register
@dataclass
class SWEBenchVerifiedAdapter(BenchmarkAdapter):
    """SWE-bench Verified adapter (heuristic offline scorer + docker hook)."""

    name: str = "swe_bench_verified"
    requires_long_context: bool = False
    license_note: str = (
        "synthetic templates for offline; "
        "real princeton-nlp/SWE-bench_Verified (CC-BY-4.0) loaded via HF"
    )

    default_n: int = 20
    load_real: bool = False
    strict_hf: bool = False
    hf_dataset_id: str = "princeton-nlp/SWE-bench_Verified"
    hf_split: str = "test"
    file_overlap_weight: float = 0.5
    line_jaccard_weight: float = 0.5
    correctness_threshold: float = 0.5

    def load_examples(self, *, n: int | None = None, seed: int = 0) -> list[BenchmarkExample]:
        n_use = n or self.default_n
        if self.load_real:
            try:
                return self._load_from_hf(n=n_use, seed=seed)
            except HFUnavailable as exc:
                if self.strict_hf:
                    raise RuntimeError(
                        f"SWE-bench Verified live loader failed with strict_hf=True: {exc}. "
                        "Refusing to fall back to synthetic under load_real=True."
                    ) from exc
                logger.warning("SWE-bench Verified real loader unavailable (%s); using synthetic", exc)
        return self._load_synthetic(n=n_use, seed=seed)

    def _load_synthetic(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        synth = generate_swe_examples(n=n, seed=seed)
        return [
            BenchmarkExample(
                id=s.instance_id,
                prompt=s.problem_statement,
                ground_truth={
                    "patch": s.gold_patch,
                    "file_path": s.file_path,
                    "fail_to_pass": s.fail_to_pass,
                    "pass_to_pass": s.pass_to_pass,
                },
                metadata={
                    "source": "synthetic",
                    "repo": s.repo,
                    "base_commit": s.base_commit,
                },
            )
            for s in synth
        ]

    def _load_from_hf(self, *, n: int, seed: int) -> list[BenchmarkExample]:
        rows = load_hf_examples(
            dataset_id=self.hf_dataset_id,
            split=self.hf_split,
            n=n,
            seed=seed,
        )
        examples: list[BenchmarkExample] = []
        for row in rows:
            instance_id = row.get("instance_id")
            problem = row.get("problem_statement", "")
            gold_patch = row.get("patch", "")
            if not instance_id or not problem or not gold_patch:
                continue
            file_paths = list(_files_in_diff(gold_patch))
            primary_file = file_paths[0] if file_paths else ""
            fail_to_pass = row.get("FAIL_TO_PASS", []) or []
            pass_to_pass = row.get("PASS_TO_PASS", []) or []
            examples.append(
                BenchmarkExample(
                    id=instance_id,
                    prompt=problem,
                    ground_truth={
                        "patch": gold_patch,
                        "file_path": primary_file,
                        "fail_to_pass": fail_to_pass,
                        "pass_to_pass": pass_to_pass,
                    },
                    metadata={
                        "source": "huggingface",
                        "hf_dataset_id": self.hf_dataset_id,
                        "repo": row.get("repo", ""),
                        "base_commit": row.get("base_commit", ""),
                        "test_patch": row.get("test_patch", ""),
                        "hints_text": row.get("hints_text", ""),
                    },
                )
            )
        if not examples:
            raise HFUnavailable("SWE-bench Verified HF rows produced 0 usable examples")
        return examples

    def render_prompt(self, example: BenchmarkExample, *, condition: str) -> str:
        gt = example.ground_truth
        assert isinstance(gt, dict)
        target_file = gt.get("file_path", "")
        if condition == "harness_on":
            return (
                "You are a careful coding agent fixing a real GitHub issue. "
                "Follow this discipline:\n"
                "  1. Re-read the issue. Identify the single most likely file to change.\n"
                "  2. Write a MINIMAL unified diff (`diff --git a/PATH b/PATH` headers).\n"
                "  3. Touch only the lines required to satisfy the failing tests.\n"
                "  4. Do not include explanatory prose outside diff comments.\n\n"
                f"## Issue\n{example.prompt}\n\n"
                f"## Likely target file\n{target_file or '(not specified)'}\n\n"
                "## Your patch (unified diff only)\n"
            )
        return (
            f"## Issue\n{example.prompt}\n\n"
            "Write a patch that resolves this issue."
        )

    def system_prompt(self, *, condition: str) -> str:
        if condition == "harness_on":
            return (
                "You produce only unified diffs that apply with `git apply`. "
                "You never invent file paths or fabricate symbols you cannot "
                "see in the issue text."
            )
        return ""

    def score(self, example: BenchmarkExample, output: ModelOutput) -> tuple[float, bool, str]:
        gt = example.ground_truth
        assert isinstance(gt, dict)
        gold_patch = gt.get("patch", "")
        if not gold_patch:
            return 0.0, False, output.text.strip()

        gold_files = _files_in_diff(gold_patch)
        pred_files = _files_in_diff(output.text)
        file_overlap = (
            len(gold_files & pred_files) / len(gold_files | pred_files)
            if (gold_files | pred_files)
            else 0.0
        )
        line_sim = _jaccard(_changed_lines(gold_patch), _changed_lines(output.text))
        composite = (
            self.file_overlap_weight * file_overlap
            + self.line_jaccard_weight * line_sim
        )
        passes = composite >= self.correctness_threshold
        return composite, passes, output.text.strip()

    def verify_with_swebench_harness(
        self,
        example: BenchmarkExample,
        output: ModelOutput,
        *,
        timeout_s: int = 600,
    ) -> bool:
        """Real verification via the official SWE-bench docker harness.

        Imports `swebench` lazily; raises SwebenchHarnessUnavailable if the
        harness or docker isn't present so the caller can fall back to the
        heuristic scorer with a clear log line. This is the contract used by
        P6-C; CI never calls it.
        """
        try:
            from swebench.harness.run_evaluation import run_instance  # type: ignore
        except ImportError as exc:
            raise SwebenchHarnessUnavailable(
                "swebench harness is not installed; "
                "`pip install swebench` and ensure docker is running"
            ) from exc

        gt = example.ground_truth
        assert isinstance(gt, dict)
        prediction = {
            "instance_id": example.id,
            "model_patch": output.text,
            "model_name_or_path": "context-engineering-harness",
        }
        try:
            result = run_instance(  # type: ignore[call-arg]
                example.metadata or {},
                prediction,
                rm_image=False,
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — surface anything to the caller as failure
            logger.warning("SWE-bench harness raised on %s: %s", example.id, exc)
            return False
        return bool(result and result.get("resolved"))
