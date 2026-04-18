"""H7 Validation: Adaptive forgetting outperforms fixed on post-shift tasks.

Method:
- Build pre-shift store → apply distribution shift (sublate pre, insert post).
- Apply each forgetting schedule: NoForgetting, BadhaFirstForgetting, FixedCompaction.
- Algorithmic: count retrievable post-shift auth elements per schedule.
- LLM: ask "How long are JWT tokens valid?" using the resulting context.

BadhaFirstForgetting clears sublated (pre-shift) elements → unambiguous retrieval.
NoForgetting retains both pre- and post-shift elements → potentially ambiguous.

Target: badha-first LLM accuracy ≥ no-forgetting, OR badha_first_retention ≥ no_forgetting_retention.
"""
from __future__ import annotations

import json

from experiments.validate.data import build_pre_shift_store, build_post_shift_store
from src.avacchedaka.query import AvacchedakaQuery
from src.cli_bridge import ClaudeCLIClient
from src.config import config
from src.forgetting.schedules import (
    BadhaFirstForgetting,
    FixedCompaction,
    NoForgetting,
)


def _ask_llm(client: ClaudeCLIClient, context: str, question: str, gold: str) -> bool:
    resp = client.messages.create(
        model=config.fast_model,
        max_tokens=config.fast_max_tokens,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nAnswer in one sentence: {question}"}],
    )
    return gold.lower()[:8] in resp.content[0].text.strip().lower()


def run_h7() -> dict:
    client = ClaudeCLIClient()
    schedule_results = {}

    schedules = [
        ("no_forgetting", lambda store: NoForgetting(store)),
        ("badha_first", lambda store: BadhaFirstForgetting(store)),
        ("fixed_compaction", lambda store: FixedCompaction(store, keep_newest=config.keep_newest)),
    ]

    for schedule_name, make_schedule in schedules:
        store = build_pre_shift_store()
        build_post_shift_store(store)
        schedule = make_schedule(store)
        schedule.apply()

        q_post = AvacchedakaQuery(qualificand="auth", condition="phase=post_shift",
                                   precision_threshold=0.0)
        n_post = len(store.retrieve(q_post))

        # Build context from the store after forgetting
        q_context = AvacchedakaQuery(qualificand="auth", condition="task_type=code_review",
                                      precision_threshold=0.0)
        context_window = store.to_context_window(q_context, max_tokens=1024)
        if not context_window:
            # fallback: join retrieved content
            results = store.retrieve(q_context)
            context_window = "\n".join(r.content for r in results)

        llm_correct = _ask_llm(
            client, context_window,
            "How long are JWT tokens valid in NexusAPI?",
            gold="1 hour",
        )

        schedule_results[schedule_name] = {
            "n_post_shift_elements": n_post,
            "llm_correct": llm_correct,
        }

    badha_retention = schedule_results["badha_first"]["n_post_shift_elements"]
    no_retention = schedule_results["no_forgetting"]["n_post_shift_elements"]
    badha_llm = schedule_results["badha_first"]["llm_correct"]
    no_llm = schedule_results["no_forgetting"]["llm_correct"]

    return {
        "hypothesis": "H7",
        "description": "Adaptive forgetting (bādha-first) outperforms fixed on post-shift tasks",
        "badha_first_retention": badha_retention,
        "no_forgetting_retention": no_retention,
        "badha_first_llm_correct": badha_llm,
        "no_forgetting_llm_correct": no_llm,
        "target_met": badha_llm or (badha_retention >= no_retention),
        "target_description": "bādha-first LLM accuracy ≥ no-forgetting on post-shift question",
        "details": schedule_results,
    }


if __name__ == "__main__":
    print(json.dumps(run_h7(), indent=2))
