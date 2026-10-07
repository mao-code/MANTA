"""Evidence-gated reflection: lessons come only from tested, stratified process evidence."""

import unittest
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.skill import (
    SkillReflector,
    evidence_signature,
    gated_process_clean,
    summary_from_candidate,
    supported_findings,
)

SEED_SKILL = (
    "# Topology Planning Skill\n\n"
    "## Standing principles\n1. one executor for writes.\n\n"
    "## How to choose a topology\n- retrieval -> searchers.\n\n"
    "## Lessons from experience\n- (none yet)\n"
)


def _revision(lesson: str) -> str:
    return SEED_SKILL.replace("- (none yet)", lesson) + ("Padding for the length guard. " * 8)


class _CountingLLM(OpenRouterLLMClient):
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def generate(self, **kwargs: Any) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=self._text,
            token_in=10,
            token_out=10,
            cost_usd=0.0,
            model="stub",
            mock_used=False,
            metadata={},
        )


def _run(benchmark: str, pattern: str, clean: bool, *, writes: bool = False) -> dict[str, Any]:
    return {
        "key": f"{benchmark}::tools::medium",
        "benchmark": benchmark,
        "pattern": pattern,
        "audit_modes": [] if clean else ["message_compaction_loss"],
        "termination_reason": "consensus_reached",
        "writes": writes,
    }


def _runs(benchmark: str, pattern: str, clean: int, total: int, **kw: Any) -> list[dict]:
    return [_run(benchmark, pattern, i < clean, **kw) for i in range(total)]


class TestEvidenceGate(unittest.TestCase):
    def test_config_defaults_to_batch_and_rejects_unknown_mode(self) -> None:
        self.assertEqual(SelfEvolvedConfig().reflection_mode, "batch")
        with self.assertRaises(ValueError):
            SelfEvolvedConfig(reflection_mode="anything").validate()

    def test_single_branch_topology_is_not_penalized_for_missing_consensus(self) -> None:
        no_consensus = {"audit_modes": [], "termination_reason": "max_rounds_reached"}
        self.assertTrue(gated_process_clean({**no_consensus, "pattern": "singleton/1"}))
        self.assertTrue(gated_process_clean({**no_consensus, "pattern": "star/2"}))
        self.assertFalse(gated_process_clean({**no_consensus, "pattern": "star/3"}))
        self.assertFalse(
            gated_process_clean(
                {**no_consensus, "pattern": "singleton/1", "audit_modes": ["branch_collapse"]}
            )
        )

    def test_signature_separates_tasks_that_write_external_state(self) -> None:
        summary = summary_from_candidate(
            {
                "key": "workbench::tools::medium",
                "state_changing_tools": True,
                "final_pattern": {"pattern": "chain", "num_agents": 2},
            }
        )
        self.assertEqual(evidence_signature(summary), "tools_with_writes::medium")
        self.assertEqual(evidence_signature({"key": "b::tools::short"}), "tools::short")

    def test_comparisons_stay_within_one_benchmark_family(self) -> None:
        # Pooled, chain/2 (A) looks far cleaner than star/3 (B). Within each family only
        # one topology was used, so there is nothing to compare.
        runs = _runs("bench_a", "chain/2", 9, 10) + _runs("bench_b", "star/3", 0, 10)
        self.assertEqual(supported_findings(runs), [])

    def test_supported_finding_needs_enough_runs_and_a_significant_gap(self) -> None:
        self.assertEqual(
            supported_findings(_runs("a", "chain/2", 4, 4) + _runs("a", "chain/3", 0, 4)), []
        )
        findings = supported_findings(_runs("a", "chain/2", 8, 8) + _runs("a", "chain/3", 0, 8))
        self.assertEqual(
            [(f["signature"], f["better"], f["worse"]) for f in findings],
            [("tools::medium", "chain/2", "chain/3")],
        )
        self.assertLess(findings[0]["p_value"], 0.05)

    def test_finding_reversed_in_another_family_is_dropped(self) -> None:
        runs = (
            _runs("a", "chain/2", 8, 8)
            + _runs("a", "star/3", 0, 8)
            + _runs("b", "chain/2", 0, 8)
            + _runs("b", "star/3", 8, 8)
        )
        self.assertEqual(supported_findings(runs), [])

    def test_gate_skips_the_llm_without_new_supported_findings(self) -> None:
        config = SelfEvolvedConfig(reflection_mode="evidence_gated")
        llm = _CountingLLM(_revision("- chain/2 ran clean 8/8, chain/3 0/8."))
        reflector = SkillReflector(llm, config)
        weak = _runs("a", "chain/2", 3, 3)

        result = reflector.reflect(current_skill=SEED_SKILL, run_summaries=weak, history=weak)
        self.assertEqual(
            (result.changed, result.reason, llm.calls), (False, "no_supported_findings", 0)
        )

        strong = _runs("a", "chain/2", 8, 8) + _runs("a", "chain/3", 0, 8)
        first = reflector.reflect(
            current_skill=SEED_SKILL, run_summaries=strong[-5:], history=strong
        )
        self.assertEqual((first.changed, llm.calls), (True, 1))
        again = reflector.reflect(
            current_skill=first.skill_markdown,
            run_summaries=strong[-5:],
            history=strong,
            previous_findings=first.findings,
        )
        self.assertEqual(
            (again.changed, again.reason, llm.calls), (False, "no_new_supported_findings", 1)
        )

    def test_gated_rewrite_rejects_unsupported_claims_and_benchmark_names(self) -> None:
        config = SelfEvolvedConfig(reflection_mode="evidence_gated")
        history = _runs("bench_a", "chain/2", 8, 8) + _runs("bench_a", "chain/3", 0, 8)
        for lesson, reason in (
            ("- star/3 runs clean on everything.", "unsupported_topology_claim"),
            ("- chain/2 beats chain/3 on bench_a.", "names_benchmark"),
            (
                "- chain/2 ran clean 8/8, chain/3 0/8.\n- Always add a validator.\n"
                "- Prefer fewer agents.",
                "unsupported_lesson",
            ),
        ):
            result = SkillReflector(_CountingLLM(_revision(lesson)), config).reflect(
                current_skill=SEED_SKILL, run_summaries=history, history=history
            )
            self.assertEqual((result.changed, result.reason), (False, reason))

    def test_batch_mode_is_unchanged(self) -> None:
        llm = _CountingLLM(_revision("- star/3 ran clean 1/1."))
        result = SkillReflector(llm, SelfEvolvedConfig()).reflect(
            current_skill=SEED_SKILL, run_summaries=_runs("a", "star/3", 1, 1)
        )
        self.assertEqual((result.changed, llm.calls), (True, 1))


if __name__ == "__main__":
    unittest.main()
