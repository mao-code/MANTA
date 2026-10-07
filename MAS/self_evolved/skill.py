"""Topology Planning Skill: the long-term playbook as an agent-maintained SKILL.md.

Unlike the structured JSON playbook (``playbook.py``), this is a long-form markdown
document — the planner loads it in full at plan time (like a skill), and an LLM
**reflection agent** rewrites it from run outcomes (``SkillReflector``).

Division of responsibility:
- read side: ``TopologySkill.load(path).prompt_section()`` returns the markdown the
  planner injects. When the file exists it is the planner's primary long-term memory;
  the JSON playbook is the deterministic fallback when it is absent.
- write side: ``SkillReflector.reflect(...)`` is given the current skill plus run
  outcomes labelled by PROCESS SIGNALS ONLY (``is_process_clean`` — auditor findings +
  decision-grade consensus, never ``benchmark.evaluate(...).success``) and returns a
  revised markdown skill. The default writer is the online ``OnlineSkillUpdater`` (every
  N runs); ``scripts/reflect_topology_skill.py`` is the offline equivalent. Keeping the
  benchmark verdict out of the skill is what stops the study from being biased.

The reflection LLM falls back to leaving the skill unchanged when mocked or unusable,
mirroring the termination/auditor judges, so offline tests stay deterministic.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SelfEvolvedConfig
from .playbook import is_process_clean

logger = logging.getLogger(__name__)


def summary_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """One run's playbook candidate → a process-only reflection summary.

    Uses **no ground truth**: the outcome label is a process proxy
    (``is_process_clean`` — the auditor flagged no failure modes and the run reached
    decision-grade consensus), so reflection never learns from the held-out benchmark
    verdict. The single source of the summary shape consumed by
    ``SkillReflector.reflect``, shared by the online updater (in-memory ``run_metadata``
    candidate) and the offline ``scripts/reflect_topology_skill.py``, so both feed the
    reflector identical rows.
    """

    final = candidate.get("final_pattern") or candidate.get("initial_pattern") or {}
    pattern = f"{final.get('pattern', '?')}/{final.get('num_agents', '?')}"
    return {
        "key": str(candidate.get("key", "")),
        "benchmark": str(candidate.get("benchmark", "")),
        "pattern": pattern,
        "process_outcome": "clean" if is_process_clean(candidate) else "flagged",
        "audit_modes": list(candidate.get("audit_modes", []) or []),
        "termination_reason": str(candidate.get("termination_reason", "")),
        "average_confidence": round(float(candidate.get("average_confidence", 0.0) or 0.0), 2),
        "writes": bool(candidate.get("state_changing_tools", False)),
    }

# -- evidence-gated reflection (self_evolved.reflection_mode = "evidence_gated") ----------
#
# The batch reflector lets the LLM read raw counts from the latest batch, so one batch
# (e.g. "chain/2 clean 8/8") can become a rule. The gated reflector instead computes, in
# code and over all runs so far, which topology comparisons are statistically supported,
# and the LLM may only put those into words.

MIN_RUNS_PER_TOPOLOGY = 5
FINDING_ALPHA = 0.05


def _single_branch(pattern: str) -> bool:
    name, _, size = str(pattern).partition("/")
    try:
        agents = int(size)
    except ValueError:
        return False
    return name == "singleton" or agents <= 1 or (name == "star" and agents <= 2)


def gated_process_clean(summary: dict[str, Any]) -> bool:
    """``is_process_clean`` without its structural bias (process signals only).

    A topology with one answering branch (a singleton, or a star with one worker) can
    never reach multi-agent consensus, so it is judged on the auditor's findings alone.
    """

    if summary.get("audit_modes"):
        return False
    if summary.get("termination_reason") == "consensus_reached":
        return True
    return _single_branch(str(summary.get("pattern", "")))


def evidence_signature(summary: dict[str, Any]) -> str:
    """General task characteristics of a run: tool access, external writes, prompt size."""

    parts = str(summary.get("key", "")).split("::")
    tools, size = (parts[-2], parts[-1]) if len(parts) >= 2 else ("unknown", "unknown")
    if tools == "tools" and summary.get("writes"):
        tools = "tools_with_writes"
    return f"{tools}::{size}"


def _fisher_upper_tail(a_clean: int, a_runs: int, b_clean: int, b_runs: int) -> float:
    """One-sided Fisher exact p-value that A runs clean more often than B."""

    clean = a_clean + b_clean
    total = math.comb(a_runs + b_runs, clean)
    upper = sum(
        math.comb(a_runs, k) * math.comb(b_runs, clean - k)
        for k in range(a_clean, min(a_runs, clean) + 1)
    )
    return upper / total


def supported_findings(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topology comparisons that pass a one-sided Fisher test, from process signals only.

    Topologies are compared only within the same source benchmark and task signature, so
    a family where nothing runs clean cannot make another family's topology look better.
    The benchmark only separates the comparisons; findings are stated by signature. A
    comparison that is supported in one family and reversed in another is dropped.
    """

    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in summaries:
        cell = counts[
            (str(row.get("benchmark", "")), evidence_signature(row), str(row.get("pattern", "?")))
        ]
        cell[0] += 1
        cell[1] += 1 if gated_process_clean(row) else 0
    strata: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(dict)
    for (benchmark, signature, pattern), cell in counts.items():
        if cell[0] >= MIN_RUNS_PER_TOPOLOGY:
            strata[(benchmark, signature)][pattern] = cell
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (_benchmark, signature), patterns in strata.items():
        for better, (b_runs, b_clean) in patterns.items():
            for worse, (w_runs, w_clean) in patterns.items():
                if better == worse or b_clean * w_runs <= w_clean * b_runs:
                    continue
                p_value = _fisher_upper_tail(b_clean, b_runs, w_clean, w_runs)
                if p_value >= FINDING_ALPHA:
                    continue
                finding = merged.setdefault(
                    (signature, better, worse),
                    {"signature": signature, "better": better, "worse": worse,
                     "better_clean": 0, "better_runs": 0, "worse_clean": 0, "worse_runs": 0,
                     "p_value": 1.0},
                )
                finding["better_clean"] += b_clean
                finding["better_runs"] += b_runs
                finding["worse_clean"] += w_clean
                finding["worse_runs"] += w_runs
                finding["p_value"] = min(finding["p_value"], round(p_value, 4))
    return [
        finding
        for key, finding in sorted(merged.items())
        if (key[0], key[2], key[1]) not in merged
    ]


def findings_fingerprint(findings: list[dict[str, Any]]) -> list[list[str]]:
    return sorted([str(f["signature"]), str(f["better"]), str(f["worse"])] for f in findings)


_PATTERN_TOKEN = re.compile(r"\b([a-z_]+/\d+)\b")


def _unsupported_change(
    new_text: str, current_skill: str, findings: list[dict[str, Any]], benchmarks: set[str]
) -> str:
    """Reason to reject a gated rewrite, or "" when every new claim is supported."""

    old = current_skill.casefold()
    new = new_text.casefold()
    for name in sorted(benchmarks):
        if name and name.casefold() in new and name.casefold() not in old:
            return "names_benchmark"
    allowed = {str(f["better"]) for f in findings} | {str(f["worse"]) for f in findings}
    added = set(_PATTERN_TOKEN.findall(new)) - set(_PATTERN_TOKEN.findall(old))
    if added - allowed:
        return "unsupported_topology_claim"
    if _lesson_count(new_text) > _lesson_count(current_skill) + len(findings):
        return "unsupported_lesson"
    return ""


def _lesson_count(skill_text: str) -> int:
    lessons = skill_text.split("## Lessons from experience", 1)
    if len(lessons) < 2:
        return 0
    section = lessons[1].split("\n## ", 1)[0]
    return sum(1 for line in section.splitlines() if line.lstrip().startswith("- "))


# Sections the reflection agent must keep (refine wording only, never delete) so a weak
# reflector cannot erase the planner's core method while growing the lessons.
PROTECTED_SECTIONS = ("## Standing principles", "## How to choose a topology")


class TopologySkill:
    """The long-term playbook rendered as an agent-maintained markdown skill."""

    def __init__(self, path: str | Path, text: str = "") -> None:
        self.path = Path(path)
        self.text = text

    @classmethod
    def load(cls, path: str | Path) -> TopologySkill:
        p = Path(path)
        if not p.exists():
            return cls(p, "")
        try:
            return cls(p, p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read topology skill at %s; treating as empty", p)
            return cls(p, "")

    def exists(self) -> bool:
        return bool(self.text.strip())

    def prompt_section(self, *, max_chars: int = 8000) -> str:
        """Bounded markdown for the planner prompt (full skill, generously capped)."""

        text = self.text.strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n…(skill truncated)…"
        return text

    def save(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        tmp.replace(self.path)


@dataclass(frozen=True)
class ReflectionResult:
    skill_markdown: str
    changed: bool
    reason: str
    llm: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)


def _strip_code_fences(text: str) -> str:
    """Drop a leading/trailing ```markdown fence if the model wrapped its output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


class SkillReflector:
    """LLM agent that revises the markdown skill from process-labelled run outcomes."""

    def __init__(self, llm_client: Any, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = llm_client
        self.se_config = se_config

    def reflect(
        self,
        *,
        current_skill: str,
        run_summaries: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        previous_findings: list[dict[str, Any]] | None = None,
    ) -> ReflectionResult:
        """Revise the skill. ``history`` (all runs so far, including this batch) and
        ``previous_findings`` (from the last applied update) are used only in
        ``evidence_gated`` mode."""

        if not run_summaries:
            return ReflectionResult(current_skill, False, "no_runs")

        findings: list[dict[str, Any]] = []
        if self.se_config.reflection_mode == "evidence_gated":
            evidence = list(history) if history is not None else list(run_summaries)
            findings = supported_findings(evidence)
            if not findings:
                return ReflectionResult(current_skill, False, "no_supported_findings")
            if previous_findings is not None and findings_fingerprint(
                findings
            ) == findings_fingerprint(previous_findings):
                return ReflectionResult(
                    current_skill, False, "no_new_supported_findings", findings=findings
                )
            prompt = self._build_gated_prompt(current_skill, findings)
        else:
            prompt = self._build_prompt(current_skill, run_summaries)
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                agent_type="general",
                task_id="skill_reflection",
                run_index=0,
                agent_id="skill_reflector",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Skill reflection failed; keeping current skill", exc_info=True)
            return ReflectionResult(current_skill, False, f"llm_error:{exc}")

        llm_payload = {
            "model": str(result.model),
            "mock_used": bool(result.mock_used),
            "token_in": int(result.token_in),
            "token_out": int(result.token_out),
            "cost_usd": float(result.cost_usd),
        }
        if bool(result.mock_used):
            return ReflectionResult(current_skill, False, "mock", llm_payload)

        text = _strip_code_fences(str(result.text or ""))
        # Guardrails: keep only a substantive doc that preserved the protected sections.
        if len(text) < 200 or not text.lstrip().startswith("#"):
            return ReflectionResult(current_skill, False, "unusable_output", llm_payload)
        if current_skill.strip():
            missing = [s for s in PROTECTED_SECTIONS if s in current_skill and s not in text]
            if missing:
                return ReflectionResult(
                    current_skill, False, "dropped_protected_section", llm_payload
                )
        if self.se_config.reflection_mode == "evidence_gated":
            benchmarks = {str(row.get("benchmark", "")) for row in (history or run_summaries)}
            rejected = _unsupported_change(text, current_skill, findings, benchmarks)
            if rejected:
                return ReflectionResult(current_skill, False, rejected, llm_payload, findings)
        return ReflectionResult(text, True, "updated", llm_payload, findings)

    # -- prompt ----------------------------------------------------------------

    @staticmethod
    def _aggregate(run_summaries: list[dict[str, Any]]) -> str:
        """Compact clean-by-(shape, pattern) table plus auditor failure-mode counts.

        Process signals only — the clean/total counts come from ``is_process_clean``,
        never from the benchmark verdict.
        """

        by_pattern: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        modes: dict[str, int] = defaultdict(int)
        for row in run_summaries:
            shape = str(row.get("key", "")) or "unknown"
            pattern = str(row.get("pattern", "?"))
            stat = by_pattern[(shape, pattern)]
            stat[0] += 1
            stat[1] += 1 if row.get("process_outcome") == "clean" else 0
            for mode in row.get("audit_modes", []) or []:
                modes[str(mode)] += 1
        lines = [
            "process outcomes by task shape and topology "
            "(clean/total — process signals only, NOT ground-truth correctness):"
        ]
        for (shape, pattern), (runs, clean) in sorted(by_pattern.items()):
            lines.append(f"  - {shape} | {pattern}: {clean}/{runs} ran clean")
        if modes:
            lines.append("process failure modes flagged by the auditor (across all runs):")
            for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {mode}: {count}")
        return "\n".join(lines)

    def _build_prompt(
        self, current_skill: str, run_summaries: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        system_msg = (
            "You maintain a 'Topology Planning Skill': a markdown document that teaches a "
            "planner how to choose a multi-agent topology for a task. You are given the "
            "current skill and outcomes from recent runs. IMPORTANT: the outcomes are "
            "labelled by PROCESS SIGNALS ONLY (whether the run's trace auditor flagged "
            "failure modes and whether it reached decision-grade consensus) — there is NO "
            "ground-truth correctness here. Revise the skill so it captures which "
            "topologies run cleanly and which trigger process failures.\n"
            "Rules:\n"
            f"- PRESERVE these sections, refining wording only, never deleting them: "
            f"{', '.join(PROTECTED_SECTIONS)}.\n"
            "- Grow the '## Lessons from experience' section: add or refine concise, "
            "actionable lessons grounded in the process signals (cite the evidence, e.g. "
            "'chain/3 ran clean 2/2 on tool-using medium retrieval; star/3 flagged "
            "duplicate_state_mutation 3x'). Speak of running cleanly / avoiding process "
            "failure modes, NOT of being 'correct' or 'right'.\n"
            "- Keep every lesson GENERAL — key it on task characteristics (task type, tools, "
            "size, state mutation, search breadth), not on benchmark-specific trivia.\n"
            "- Prefer revising an existing lesson over duplicating it; drop lessons the new "
            "evidence contradicts. Keep the document tight and readable.\n"
            "- Output the COMPLETE updated markdown document and nothing else (no fences)."
        )
        user_msg = (
            "## Current skill\n"
            f"{current_skill.strip() or '(empty — create the document)'}\n\n"
            "## Recent run outcomes (process signals only — no ground truth)\n"
            f"{self._aggregate(run_summaries)}\n\n"
            "Return the full updated skill markdown."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


    def _build_gated_prompt(
        self, current_skill: str, findings: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        system_msg = (
            "You maintain a 'Topology Planning Skill': a markdown document that teaches a "
            "planner how to choose a multi-agent topology for a task. You are given the "
            "current skill and a list of SUPPORTED FINDINGS. Each finding compares two "
            "topologies on tasks with the same characteristics, over all runs so far, and "
            "passed a statistical test (one-sided Fisher exact test, p < "
            f"{FINDING_ALPHA}, at least {MIN_RUNS_PER_TOPOLOGY} runs per topology). The "
            "outcomes are PROCESS SIGNALS ONLY (no auditor failure modes, and decision-grade "
            "consensus where the topology has more than one answering branch). There is NO "
            "ground-truth correctness here.\n"
            "Rules:\n"
            f"- PRESERVE these sections, refining wording only, never deleting them: "
            f"{', '.join(PROTECTED_SECTIONS)}.\n"
            "- In '## Lessons from experience', add or revise a lesson ONLY to state a "
            "supported finding, citing its counts. State no other topology comparison, and "
            "do not generalize a finding beyond the task characteristics it names.\n"
            "- Task characteristics: 'tools' = tool access; 'tools_with_writes' = tools that "
            "change external state; 'no_tools' = no tool access; 'short', 'medium', "
            "'long' = prompt length. Describe them in words. Never name a benchmark or "
            "dataset.\n"
            "- Weaken or remove an existing lesson only when a supported finding contradicts "
            "it. Otherwise keep the document unchanged. Speak of running cleanly, not of "
            "being correct.\n"
            "- Output the COMPLETE updated markdown document and nothing else (no fences)."
        )
        lines = [
            f"  - {f['signature']}: {f['better']} ran clean {f['better_clean']}/"
            f"{f['better_runs']}, {f['worse']} ran clean {f['worse_clean']}/{f['worse_runs']} "
            f"(p = {f['p_value']})"
            for f in findings
        ]
        user_msg = (
            "## Current skill\n"
            f"{current_skill.strip() or '(empty — create the document)'}\n\n"
            "## Supported findings (process signals only — no ground truth)\n"
            + "\n".join(lines)
            + "\n\nReturn the full updated skill markdown."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


class OnlineSkillUpdater:
    """Online (in-experiment) writer for the long-term skill.

    Accumulates per-run playbook candidates (process signals only — no ground truth)
    from freshly executed self_evolved runs and, every ``batch_size`` runs, reflects
    that batch into the skill markdown and fires ``on_update`` so the planner reloads
    the revised skill for the rest of the experiment. Because the experiment loop is single-threaded, ``record`` blocks
    on the reflection LLM call at a batch boundary — the run loop pauses, the skill
    updates, then the loop resumes. ``batch_size <= 0`` makes every method a no-op,
    preserving the parallel-safe, post-hoc-only default.

    A trailing partial batch (< ``batch_size`` runs) is intentionally left for the
    post-hoc ``scripts/reflect_topology_skill.py`` to catch — online updates fire only
    on exact batch boundaries.
    """

    def __init__(
        self,
        *,
        reflector: SkillReflector,
        skill_path: str | Path,
        batch_size: int,
        on_update: Callable[[], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.reflector = reflector
        self.skill_path = Path(skill_path)
        self.batch_size = int(batch_size)
        self._on_update = on_update
        self._log = log or (lambda _message: None)
        self._buffer: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._findings: list[dict[str, Any]] | None = None
        self.updates_applied = 0

    @property
    def enabled(self) -> bool:
        return self.batch_size > 0

    def record(self, candidate: dict[str, Any] | None) -> None:
        """Buffer one run's process-signal outcome; flush when the batch is full.

        Takes no ground-truth label — the reflection input is process-only
        (``summary_from_candidate``)."""

        if not self.enabled or not candidate:
            return
        self._buffer.append(summary_from_candidate(candidate))
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        batch = self._buffer
        self._buffer = []
        if not batch:
            return
        # Re-read the on-disk skill so reflection always builds on the latest text.
        skill = TopologySkill.load(self.skill_path)
        clean = sum(1 for row in batch if row.get("process_outcome") == "clean")
        self._history.extend(batch)
        result = self.reflector.reflect(
            current_skill=skill.text,
            run_summaries=batch,
            history=list(self._history),
            previous_findings=self._findings,
        )
        if result.changed:
            self._findings = list(result.findings)
            skill.save(result.skill_markdown)
            self.updates_applied += 1
            if self._on_update is not None:
                self._on_update()
            self._log(
                f"SKILL_UPDATE runs={len(batch)} clean={clean} "
                f"reason={result.reason} path={self.skill_path}"
            )
        else:
            self._log(
                f"SKILL_UPDATE_SKIP runs={len(batch)} clean={clean} reason={result.reason}"
            )
