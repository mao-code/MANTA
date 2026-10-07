"""Topology Playbook: persistent and in-run memory of topology decisions.

The playbook is a JSON file keyed by benchmark plus coarse deterministic task
features (no embeddings — keys stay auditable).

There are two memory horizons:

- short-term, turn-level memory lives only inside one run and is updated after
  each audited turn from process signals;
- long-term, cross-run memory keyed by task shape, updated every N runs (online)
  by the skill reflection agent, or offline by ``scripts/update_topology_playbook.py``.

**Both horizons use process signals only — never the benchmark verdict.** Ranking the
long-term playbook by ``benchmark.evaluate(...).success`` would leak the held-out label
into the system under study, so the playbook ranks topologies by a *process proxy*
(``is_process_clean``: the auditor flagged no failure modes and the run reached
decision-grade consensus). ``benchmark.evaluate(...).success`` remains the sole authority
for scoring/metrics, but it is kept out of the planner's memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLAYBOOK_VERSION = 1
_MAX_FAILURE_MEMORIES = 8
_MAX_ENTRIES = 200
_MAX_SHORT_TERM_ENTRIES = 8


def task_key(benchmark_name: str, task: Any, *, tools_available: bool) -> str:
    """Deterministic coarse task key: benchmark + tool availability + prompt size."""

    prompt = str(getattr(task, "prompt", "") or "")
    if len(prompt) < 400:
        size = "short"
    elif len(prompt) < 1600:
        size = "medium"
    else:
        size = "long"
    tools = "tools" if tools_available else "no_tools"
    return f"{benchmark_name or 'unknown'}::{tools}::{size}"


def is_process_clean(candidate: dict[str, Any]) -> bool:
    """Process-only quality proxy (NO ground truth).

    A run is "clean" when the Trace Auditor flagged no failure modes and the run
    reached decision-grade consensus. This is what the long-term playbook ranks
    topologies by, so the self-evolved system never learns from the held-out
    ``benchmark.evaluate(...).success`` verdict it is being measured on.
    """

    audit_modes = candidate.get("audit_modes") or []
    reason = str(candidate.get("termination_reason", ""))
    return not audit_modes and reason == "consensus_reached"


def pattern_sketch(spec_payload: dict[str, Any]) -> dict[str, Any]:
    """Compact plan-DSL-shaped sketch of a TopologySpec payload."""

    groups = spec_payload.get("groups", [])
    root_id = spec_payload.get("root_group_id", "")
    root = next((g for g in groups if g.get("group_id") == root_id), {})
    expansions = [
        {
            "parent_agent_id": group.get("parent_agent_id"),
            "pattern": group.get("pattern"),
            "num_subagents": len(group.get("member_ids", [])),
        }
        for group in groups
        if group.get("parent_agent_id")
    ]
    return {
        "pattern": root.get("pattern", ""),
        "num_agents": len(spec_payload.get("agents", [])),
        "expansions": expansions,
    }


class TopologyPlaybook:
    def __init__(self, path: str | Path, payload: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        payload = payload or {"version": PLAYBOOK_VERSION, "entries": []}
        self.version = int(payload.get("version", PLAYBOOK_VERSION))
        entries = payload.get("entries", [])
        self.entries: list[dict[str, Any]] = [
            dict(entry) for entry in entries if isinstance(entry, dict)
        ]
        # Benchmark-agnostic priors always surfaced to the planner (cross-cutting
        # laws like "state-mutating tasks use a single executor"). Unlike entries,
        # these are not keyed and transfer to benchmarks the playbook has never seen.
        principles = payload.get("principles", [])
        self.principles: list[str] = [
            str(item).strip() for item in principles if isinstance(item, str) and str(item).strip()
        ]

    @classmethod
    def load(cls, path: str | Path) -> TopologyPlaybook:
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("playbook root must be a JSON object")
            return cls(path, payload)
        except Exception:
            logger.warning("Failed to load playbook at %s; using empty", path, exc_info=True)
            return cls(path)

    def save(self) -> None:
        payload = {
            "version": self.version,
            "principles": self.principles,
            "entries": self.entries[:_MAX_ENTRIES],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    # -- read side (used by the planner during runs) --------------------------

    def global_principles(self, *, limit: int = 6) -> list[str]:
        """Benchmark-agnostic priors injected into every planner prompt."""

        return list(self.principles[:limit])

    def lookup(self, benchmark_name: str, key: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve historical experience for a task, most specific first.

        Three tiers, so a lesson learned anywhere transfers by task *shape* — this is
        what makes the long-term playbook act like a reusable skill rather than a
        per-benchmark table:
        1. exact key (this benchmark + tool availability + prompt size);
        2. other keys from the same benchmark;
        3. the same task shape (``tools::size`` suffix) learned on *other* benchmarks.

        Tier 3 is the transfer channel: a topology that worked for tool-using medium
        tasks on one benchmark is offered as experience for a tool-using medium task the
        playbook has never seen on its own benchmark.
        """

        suffix = key.split("::", 1)[1] if "::" in key else key
        exact = [entry for entry in self.entries if entry.get("key") == key]
        same_benchmark = [
            entry
            for entry in self.entries
            if entry.get("benchmark") == benchmark_name and entry.get("key") != key
        ]
        same_shape = [
            entry
            for entry in self.entries
            if entry.get("benchmark") != benchmark_name
            and str(entry.get("key", "")).split("::", 1)[-1] == suffix
        ]
        return (exact + same_benchmark + same_shape)[:limit]

    @staticmethod
    def planner_view(entry: dict[str, Any]) -> dict[str, Any]:
        """Bounded shape fed into planner prompts."""

        scope = str(entry.get("scope", "long_term"))
        support = entry.get("support", {})
        runs = int(support.get("runs", 0))
        # Process proxy, not ground truth; tolerate legacy "successes"-keyed entries.
        clean = int(support.get("clean", support.get("successes", 0)))
        if scope == "short_term":
            support_text = "turn-level process memory"
        else:
            support_text = f"{clean}/{runs} ran clean"
        memories = entry.get("failure_memories", [])
        mutation_notes = "; ".join(
            f"{memory.get('mode')}->{memory.get('mutation')}({memory.get('outcome')})"
            for memory in memories[:3]
            if isinstance(memory, dict)
        )
        return {
            "scope": scope,
            "key": str(entry.get("key", "")),
            "turn_index": entry.get("turn_index"),
            "pattern": entry.get("pattern", {}),
            "support": support_text,
            "notes": str(entry.get("notes", ""))[:200],
            "mutation": mutation_notes,
        }

    # -- write side (deterministic merge; process signals only, no ground truth) ----

    def merge_candidate(self, candidate: dict[str, Any]) -> None:
        key = str(candidate.get("key", "")).strip()
        if not key:
            return
        clean = is_process_clean(candidate)
        entry = next((item for item in self.entries if item.get("key") == key), None)
        if entry is None:
            entry = {
                "key": key,
                "benchmark": str(candidate.get("benchmark", "")),
                "pattern": dict(candidate.get("initial_pattern", {})),
                "support": {"runs": 0, "clean": 0},
                "notes": "",
                "failure_memories": [],
            }
            self.entries.append(entry)

        support = entry.setdefault("support", {"runs": 0, "clean": 0})
        support["runs"] = int(support.get("runs", 0)) + 1
        support["clean"] = int(support.get("clean", support.get("successes", 0))) + int(clean)

        # Learn the winning topology per key instead of freezing the first-seen pattern:
        # tally the PROCESS proxy (clean runs) by the pattern that actually ran
        # (post-repair) and recommend the cleanest one (highest clean rate, ties broken by
        # support). This lets the planner improve from prior experiments without ever
        # seeing the held-out benchmark verdict.
        ran_pattern = candidate.get("final_pattern") or candidate.get("initial_pattern") or {}
        if ran_pattern:
            stats = entry.setdefault("pattern_stats", {})
            pat_key = json.dumps(ran_pattern, sort_keys=True)
            stat = stats.setdefault(pat_key, {"pattern": dict(ran_pattern), "runs": 0, "clean": 0})
            stat["runs"] = int(stat.get("runs", 0)) + 1
            stat["clean"] = int(stat.get("clean", 0)) + int(clean)
            ranked = sorted(
                stats.values(),
                key=lambda s: (
                    int(s.get("clean", 0)) / max(int(s.get("runs", 0)), 1),
                    int(s.get("runs", 0)),
                ),
                reverse=True,
            )
            best = ranked[0]
            entry["pattern"] = dict(best["pattern"])

            # Distill a concise, reusable rule from the accumulated stats so the planner
            # reads an actionable learned insight (which pattern runs cleanest, which to
            # avoid) rather than a bare pattern. This is the long-term "experience" channel.
            def _sketch(stat: dict[str, Any]) -> str:
                pat = stat.get("pattern", {})
                return f"{pat.get('pattern', '?')}/{pat.get('num_agents', '?')}"

            note = f"best (cleanest): {_sketch(best)} ({best.get('clean', 0)}/{best.get('runs', 0)} clean)"
            worst = ranked[-1]
            if (
                worst is not best
                and int(worst.get("clean", 0)) == 0
                and int(worst.get("runs", 0)) > 0
            ):
                note += f"; avoid: {_sketch(worst)} (0/{worst.get('runs', 0)} clean)"
            entry["notes"] = note

        mutation = candidate.get("mutation_summary")
        if mutation:
            memories = entry.setdefault("failure_memories", [])
            memories.append(
                {
                    "mode": ",".join(candidate.get("audit_modes", [])) or "unknown",
                    "mutation": str(mutation),
                    "outcome": "clean" if clean else "flagged",
                    "termination_reason": str(candidate.get("termination_reason", "")),
                }
            )
            del memories[:-_MAX_FAILURE_MEMORIES]


class PlaybookMaintainer:
    """Builds the in-run update candidate (process signals only, no ground truth)."""

    @staticmethod
    def build_update_candidate(
        *,
        key: str,
        benchmark_name: str,
        spec_versions: list[dict[str, Any]],
        audit_reports: list[dict[str, Any]],
        mutation_payload: dict[str, Any] | None,
        mutation_payloads: list[dict[str, Any]],
        termination_decision: dict[str, Any],
        state_changing: bool = False,
    ) -> dict[str, Any]:
        audit_modes = sorted(
            {
                str(mode.get("mode", ""))
                for report in audit_reports
                for mode in report.get("detected_modes", [])
                if mode.get("mode")
            }
        )
        mutation_summaries: list[str] = []
        history = list(mutation_payloads or ([mutation_payload] if mutation_payload else []))
        for payload in history:
            if not payload.get("mutation"):
                continue
            ops = payload["mutation"].get("ops", [])
            summary = ",".join(
                f"{op.get('op')}:{op.get('args', {}).get('pattern', '')}".rstrip(":")
                for op in ops
                if isinstance(op, dict)
            )
            if summary:
                mutation_summaries.append(summary)
        mutation_summary = " -> ".join(mutation_summaries)
        return {
            "key": key,
            "benchmark": benchmark_name,
            "initial_pattern": pattern_sketch(spec_versions[0]) if spec_versions else {},
            "final_pattern": pattern_sketch(spec_versions[-1]) if spec_versions else {},
            "mutated": len(spec_versions) > 1,
            "mutation_summary": mutation_summary,
            "audit_modes": audit_modes,
            "termination_reason": str(termination_decision.get("reason", "")),
            "consensus_ratio": float(termination_decision.get("consensus_ratio", 0.0)),
            "average_confidence": float(termination_decision.get("average_confidence", 0.0)),
            "state_changing_tools": bool(state_changing),
        }


class ShortTermTopologyPlaybook:
    """In-memory turn-level playbook for one self-evolved run.

    It records process observations before benchmark correctness is known. The
    entries are planner priors for the next in-run repair decision, not
    persistent success evidence.
    """

    def __init__(self, *, max_entries: int = _MAX_SHORT_TERM_ENTRIES) -> None:
        self.max_entries = max(1, int(max_entries))
        self.entries: list[dict[str, Any]] = []

    def record_turn(
        self,
        *,
        key: str,
        benchmark_name: str,
        turn_index: int,
        spec_payload: dict[str, Any],
        audit_report: dict[str, Any] | None,
        termination_decision: dict[str, Any],
    ) -> dict[str, Any]:
        audit_report = audit_report or {}
        detected_modes = [
            str(mode.get("mode", ""))
            for mode in audit_report.get("detected_modes", [])
            if isinstance(mode, dict) and mode.get("mode")
        ]
        repair_recommended = bool(audit_report.get("repair_recommended", False))
        recommendation = str(audit_report.get("recommendation", "")).strip()
        reason = str(termination_decision.get("reason", "")).strip()
        notes_parts = [
            f"turn={int(turn_index)}",
            f"repair_recommended={repair_recommended}",
            f"modes={','.join(detected_modes) if detected_modes else 'none'}",
        ]
        if reason:
            notes_parts.append(f"decision={reason}")
        if recommendation:
            notes_parts.append(recommendation)

        entry = {
            "scope": "short_term",
            "key": key,
            "benchmark": benchmark_name,
            "turn_index": int(turn_index),
            "spec_version": int(spec_payload.get("version", 0)),
            "pattern": pattern_sketch(spec_payload),
            "support": {"kind": "process_signal", "runs": 1, "clean": 0},
            "notes": "; ".join(notes_parts)[:400],
            "repair_recommended": repair_recommended,
            "termination_reason": reason,
            "failure_memories": [
                {
                    "mode": mode,
                    "mutation": "pending_trace_backed_repair"
                    if repair_recommended
                    else "no_repair",
                    "outcome": "process_signal",
                }
                for mode in detected_modes[:3]
            ],
        }
        self.entries.append(entry)
        del self.entries[: -self.max_entries]
        return entry

    def planner_entries(self) -> list[dict[str, Any]]:
        return [TopologyPlaybook.planner_view(entry) for entry in self.entries]

    def to_payload(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]
