import json
import tempfile
import unittest
from pathlib import Path

from benchmark.browsecomp import (
    BrowseCompBenchmark,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
)


class TestBrowseCompBenchmark(unittest.TestCase):
    def test_load_and_evaluate_substring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            decrypted = base / "decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"

            row = {
                "query_id": "q1",
                "query": "Which doc contains answer?",
                "answer": "42",
                "gold_docs": [{"docid": "100", "text": "gold", "url": "u"}],
                "evidence_docs": [{"docid": "200", "text": "ev", "url": "u"}],
            }
            decrypted.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 200 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

            bench = BrowseCompBenchmark(
                {
                    "decrypted_path": str(decrypted),
                    "qrel_evidence_path": str(qrel_evidence),
                    "qrel_golds_path": str(qrel_golds),
                    "eval_mode": "substring",
                }
            )
            tasks = bench.load_tasks()
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.task_id, "q1")

            # Good prediction
            eval_ok = bench.evaluate(
                task,
                "Final answer: 42",
                run_metadata={"retrieved_docids": ["100", "200"]},
            )
            self.assertTrue(eval_ok.success)
            self.assertEqual(eval_ok.details["retrieval"]["recall_evidence"], 1.0)
            self.assertEqual(eval_ok.details["retrieval"]["recall_gold"], 1.0)

            # Bad prediction
            eval_bad = bench.evaluate(
                task,
                "Final answer: 0",
                run_metadata={"retrieved_docids": []},
            )
            self.assertFalse(eval_bad.success)
            self.assertEqual(eval_bad.score, 0.0)

    def test_get_document_tool_returns_full_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            decrypted = base / "decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"

            full_text = " ".join(f"token{i}" for i in range(20))
            row = {
                "query_id": "q1",
                "query": "Which doc contains answer?",
                "answer": "token19",
                "gold_docs": [{"docid": "100", "text": full_text, "url": "u"}],
                "evidence_docs": [],
            }
            decrypted.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

            bench = BrowseCompBenchmark(
                {
                    "decrypted_path": str(decrypted),
                    "qrel_evidence_path": str(qrel_evidence),
                    "qrel_golds_path": str(qrel_golds),
                    "tool_snippet_max_tokens": 3,
                    "include_get_document": True,
                }
            )
            task = bench.load_tasks()[0]
            tools = {tool["name"]: tool for tool in bench._build_tools_for_task(task.task_id)}

            search_result = tools["search"]["handler"]({"query": "token19"})[0]
            document = tools["get_document"]["handler"]({"docid": "100"})

            self.assertEqual(search_result["snippet"], "token17 token18 token19")
            self.assertEqual(document["snippet"], "token0 token1 token2")
            self.assertEqual(document["text"], full_text)

    def test_search_ignores_prompt_stopwords_and_zero_score_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            decrypted = base / "decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"

            row = {
                "query_id": "q1",
                "query": "Which institution gathered alpine moss samples?",
                "answer": "Needle College",
                "gold_docs": [
                    {
                        "docid": "gold",
                        "text": "Needle College botany students gathered alpine moss samples near the ridge.",
                        "url": "u",
                    }
                ],
                "evidence_docs": [
                    {
                        "docid": "generic",
                        "text": "The institution was in the city and displayed moss at an event.",
                        "url": "u",
                    }
                ],
            }
            decrypted.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 gold 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 gold 1\n", encoding="utf-8")

            bench = BrowseCompBenchmark(
                {
                    "decrypted_path": str(decrypted),
                    "qrel_evidence_path": str(qrel_evidence),
                    "qrel_golds_path": str(qrel_golds),
                    "tool_snippet_max_tokens": 6,
                    "include_get_document": True,
                }
            )
            task = bench.load_tasks()[0]
            search = {
                tool["name"]: tool for tool in bench._build_tools_for_task(task.task_id)
            }["search"]["handler"]

            results = search({"query": "Please tell me the institution with alpine moss samples"})
            self.assertEqual(results[0]["docid"], "gold")
            self.assertIn("alpine moss samples", results[0]["snippet"])
            self.assertEqual(search({"query": "zzzxxyy unlikelytoken"}), [])

            expanded = search({"query": "moss", "k": 2})
            self.assertEqual(len(expanded), 2)

    def test_substring_normalization_tolerates_accents_and_hyphens(self) -> None:
        self.assertTrue(
            BrowseCompBenchmark._normalized_exact_or_numeric_match(
                "The answer is Laura Lojo Rodríguez.",
                "Laura Lojo-Rodriguez",
            )
        )


class TestParseJudgeResponse(unittest.TestCase):
    def test_parse_standard_format(self) -> None:
        response = (
            "extracted_final_answer: Paris\n\n"
            "reasoning: The answer matches the correct answer.\n\n"
            "correct: yes\n\n"
            "confidence: 95%"
        )
        result = parse_judge_response(response)
        self.assertEqual(result["extracted_final_answer"], "Paris")
        self.assertTrue(result["correct"])
        self.assertEqual(result["confidence"], 95.0)
        self.assertFalse(result["parse_error"])

    def test_parse_bold_format(self) -> None:
        response = (
            "**extracted_final_answer:** Tokyo\n"
            "**reasoning:** Matches.\n"
            "**correct:** no\n"
            "**confidence:** 30"
        )
        result = parse_judge_response(response)
        self.assertEqual(result["extracted_final_answer"], "Tokyo")
        self.assertFalse(result["correct"])
        self.assertEqual(result["confidence"], 30.0)

    def test_parse_empty(self) -> None:
        result = parse_judge_response("")
        self.assertTrue(result["parse_error"])

    def test_parse_missing_correct(self) -> None:
        response = "extracted_final_answer: something\nreasoning: blah"
        result = parse_judge_response(response)
        self.assertTrue(result["parse_error"])
        self.assertIsNone(result["correct"])


class TestCitations(unittest.TestCase):
    def test_extract_half_width(self) -> None:
        text = "The answer is X [123] and Y [456, 789]."
        docids = extract_citations_from_response(text)
        self.assertIn("123", docids)
        self.assertIn("456", docids)
        self.assertIn("789", docids)

    def test_extract_fullwidth(self) -> None:
        text = "回答是 X【100】。"
        docids = extract_citations_from_response(text)
        self.assertIn("100", docids)

    def test_empty(self) -> None:
        self.assertEqual(extract_citations_from_response(""), [])

    def test_citation_metrics(self) -> None:
        metrics = compute_citation_metrics(["1", "2", "3"], ["2", "3", "4"])
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)

    def test_citation_metrics_empty(self) -> None:
        metrics = compute_citation_metrics([], ["1", "2"])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_restricted_load_keeps_only_requested_tasks_with_same_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            decrypted = Path(tmpdir) / "decrypted.jsonl"
            rows = [
                {
                    "query_id": qid,
                    "query": f"question {qid}",
                    "answer": qid,
                    "gold_docs": [{"docid": f"g{qid}", "text": f"gold {qid}", "url": "u"}],
                }
                for qid in ("q1", "q2", "q3")
            ]
            decrypted.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
            config = {"decrypted_path": str(decrypted), "eval_mode": "substring"}

            full = BrowseCompBenchmark(config)
            full.load_tasks()
            restricted = BrowseCompBenchmark(config)
            restricted.restrict_to_task_ids({"q2"})
            tasks = restricted.load_tasks()

            self.assertEqual([task.task_id for task in tasks], ["q2"])
            self.assertEqual(list(restricted._task_docs), ["q2"])
            self.assertEqual(restricted._task_docs["q2"], full._task_docs["q2"])


if __name__ == "__main__":
    unittest.main()
