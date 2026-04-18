"""In-process client for the `pratyaksha-context-eng-harness` plugin.

The plugin is shipped as an MCP server that runs in a subprocess and is
called by Claude Code over JSON-RPC. For end-to-end validation runs we
need the *exact same code path* without paying the IPC cost — so this
module imports the plugin's `mcp/server.py` once, snapshots its module
state on entry, and exposes typed wrappers around every tool function.

Why "in-process"?

  * Reproducibility: each test run gets a fresh `_State` snapshot.
  * Fidelity: the wrappers call the same `context_insert`,
    `sublate_with_evidence`, `boundary_compact`, etc. functions Claude
    Code's MCP runtime would call — there is no parallel implementation
    of plugin behaviour anywhere in this repo.
  * Speed: we run thousands of trials per hypothesis without spawning
    one MCP subprocess per call.

If anyone ever modifies the plugin's tool behaviour, the H3/H4/H5
re-runs that depend on this client will see the change immediately.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLUGIN_SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugin"
    / "pratyaksha-context-eng-harness"
    / "mcp"
    / "server.py"
)


_SERVER_MODULE = None


def _load_plugin_server():
    """Load the plugin server module once, cache it for subsequent calls."""
    global _SERVER_MODULE
    if _SERVER_MODULE is not None:
        return _SERVER_MODULE
    if not PLUGIN_SERVER_PATH.exists():
        raise RuntimeError(f"plugin server not found at {PLUGIN_SERVER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "_pratyaksha_inproc", str(PLUGIN_SERVER_PATH)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pratyaksha_inproc"] = module
    spec.loader.exec_module(module)
    _SERVER_MODULE = module
    logger.debug("loaded pratyaksha plugin in-process from %s", PLUGIN_SERVER_PATH)
    return module


class PratyakshaPluginClient:
    """Typed in-process facade over the plugin's MCP tool functions.

    Each instance gets its own `_State` snapshot — call `reset()` between
    trials so one experiment cannot leak into the next.
    """

    def __init__(self) -> None:
        self._mod = _load_plugin_server()
        self.reset()

    # --- lifecycle -----------------------------------------------------

    def reset(self) -> None:
        """Clear every per-session attribute on the plugin's STATE singleton."""
        new_state = self._mod._State()
        # Mutate in-place to keep the module-level binding intact.
        self._mod.STATE.elements = new_state.elements
        self._mod.STATE.sakshi = new_state.sakshi
        self._mod.STATE.budget_total = new_state.budget_total
        self._mod.STATE.budget_used = new_state.budget_used

    # --- Avacchedaka store --------------------------------------------

    def insert(
        self,
        *,
        id: str,
        content: str,
        precision: float,
        qualificand: str,
        qualifier: str,
        condition: str,
        relation: str = "inherence",
        provenance: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return self._mod.context_insert(
            self._mod.InsertInput(
                id=id,
                content=content,
                precision=precision,
                qualificand=qualificand,
                qualifier=qualifier,
                condition=condition,
                relation=relation,
                provenance=provenance,
                overwrite=overwrite,
            )
        )

    def retrieve(
        self,
        *,
        qualificand: str,
        condition: str = "",
        precision_threshold: float = 0.5,
        max_elements: int = 20,
    ) -> dict[str, Any]:
        return self._mod.context_retrieve(
            self._mod.RetrieveInput(
                qualificand=qualificand,
                condition=condition,
                precision_threshold=precision_threshold,
                max_elements=max_elements,
            )
        )

    def get(self, element_id: str) -> dict[str, Any]:
        return self._mod.context_get(self._mod.GetByIdInput(element_id=element_id))

    def sublate(self, element_id: str, by_element_id: str) -> dict[str, Any]:
        return self._mod.context_sublate(
            self._mod.SublateInput(element_id=element_id, by_element_id=by_element_id)
        )

    def list_qualificands(self) -> dict[str, Any]:
        return self._mod.list_qualificands()

    # --- Sublation with evidence --------------------------------------

    def sublate_with_evidence(
        self,
        *,
        older_id: str,
        newer_content: str,
        newer_precision: float,
        qualificand: str,
        qualifier: str,
        condition: str,
        provenance: str = "",
    ) -> dict[str, Any]:
        return self._mod.sublate_with_evidence(
            self._mod.SublateWithEvidenceInput(
                older_id=older_id,
                newer_content=newer_content,
                newer_precision=newer_precision,
                qualificand=qualificand,
                qualifier=qualifier,
                condition=condition,
                provenance=provenance,
            )
        )

    def detect_conflict(
        self, *, qualificand: str, condition: str = "", precision_threshold: float = 0.5
    ) -> dict[str, Any]:
        return self._mod.detect_conflict(
            self._mod.RetrieveInput(
                qualificand=qualificand,
                condition=condition,
                precision_threshold=precision_threshold,
                max_elements=200,
            )
        )

    # --- Compaction ---------------------------------------------------

    def compact(
        self,
        *,
        precision_threshold: float = 0.3,
        qualificand: str = "",
        task_context: str = "",
    ) -> dict[str, Any]:
        return self._mod.compact(
            self._mod.CompactInput(
                precision_threshold=precision_threshold,
                qualificand=qualificand,
                task_context=task_context,
            )
        )

    def boundary_compact(
        self, *, text_window: str, threshold_z: float = 2.0, precision_threshold: float = 0.3
    ) -> dict[str, Any]:
        return self._mod.boundary_compact(
            self._mod.BoundaryCompactInput(
                text_window=text_window,
                threshold_z=threshold_z,
                precision_threshold=precision_threshold,
            )
        )

    def context_window(
        self, *, qualificand: str, condition: str = "",
        max_tokens: int = 4096, precision_threshold: float = 0.5,
    ) -> dict[str, Any]:
        return self._mod.context_window(
            self._mod.ContextWindowInput(
                qualificand=qualificand,
                condition=condition,
                max_tokens=max_tokens,
                precision_threshold=precision_threshold,
            )
        )

    # --- Witness ------------------------------------------------------

    def set_sakshi(self, content: str) -> dict[str, Any]:
        return self._mod.set_sakshi(self._mod.SetSakshiInput(content=content))

    def get_sakshi(self) -> dict[str, Any]:
        return self._mod.get_sakshi()

    # --- Khyativada ---------------------------------------------------

    def classify_khyativada(
        self, *, claim: str, ground_truth: str, context: str = ""
    ) -> dict[str, Any]:
        return self._mod.classify_khyativada(
            self._mod.ClassifyInput(claim=claim, ground_truth=ground_truth, context=context)
        )

    # --- Budget -------------------------------------------------------

    def budget_status(self, last_n: int = 20) -> dict[str, Any]:
        return self._mod.budget_status(self._mod.BudgetStatusInput(last_n=last_n))

    def budget_record(self, *, tokens: int, model: str = "", note: str = "") -> dict[str, Any]:
        return self._mod.budget_record(
            self._mod.BudgetRecordInput(tokens=tokens, model=model, note=note)
        )

    # --- introspection (state inspection for assertions in tests) -----

    @property
    def state_size(self) -> int:
        return len(self._mod.STATE.elements)

    @property
    def n_active(self) -> int:
        return sum(
            1 for e in self._mod.STATE.elements.values() if e.sublated_by is None
        )

    @property
    def n_sublated(self) -> int:
        return sum(
            1 for e in self._mod.STATE.elements.values() if e.sublated_by is not None
        )
