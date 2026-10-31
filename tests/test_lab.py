import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from eval_lab import Lab, compare


CASES = [{"id":"a", "input":"input A", "checks":{"exact":"ok", "contains":["o"]}},
         {"id":"b", "input":"input B", "checks":{"json_fields":{"count":1}}}]


class LabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "eval.db"
        self.tick = 0
        def timer():
            self.tick += 0.25
            return self.tick
        self.lab = Lab(self.path, timer=timer)
        self.prompt = self.lab.add_prompt("demo", "Task {{input}}")

    def adapter(self, prompt, case):
        return {"output": "ok" if case == "a" else '{"count":1}', "input_tokens":100, "output_tokens":50}

    def run_fixture(self, **kwargs):
        return self.lab.report(self.lab.run(self.prompt["id"], copy.deepcopy(CASES), self.adapter, **kwargs))

    def test_immutable_prompt_versions_and_concurrency(self):
        def add(_):
            return self.lab.add_prompt("demo", "New {{input}}")
        with ThreadPoolExecutor(max_workers=6) as pool:
            versions = list(pool.map(add, range(12)))
        self.assertEqual(sorted(v["version"] for v in versions), list(range(2,14)))
        self.assertEqual(self.run_fixture()["snapshot"]["prompt"]["template"], "Task {{input}}")

    def test_reproducible_fixture_scores(self):
        a, b = self.run_fixture(), self.run_fixture()
        self.assertEqual(a["snapshot"]["fixture_hash"], b["snapshot"]["fixture_hash"])
        self.assertEqual(a["results"], b["results"])
        self.assertEqual(a["summary"]["mean_score"], 1)
        self.assertEqual(a["summary"]["mean_success_seconds"], .25)

    def test_exact_decimal_cost_and_token_accounting(self):
        report = self.run_fixture(input_per_1000="0.003", output_per_1000="0.006")
        self.assertEqual(Decimal(report["summary"]["known_cost"]), Decimal("0.0012"))
        self.assertTrue(report["summary"]["cost_complete"])

    def test_partial_failure_isolated_and_cost_unknown(self):
        def adapter(prompt, case):
            if case == "a":
                raise RuntimeError("secret must not appear")
            return self.adapter(prompt, case)
        report = self.lab.report(self.lab.run(self.prompt["id"], CASES, adapter))
        self.assertEqual(report["state"], "completed_with_errors")
        self.assertEqual(report["summary"]["mean_score"], .5)
        self.assertFalse(report["summary"]["cost_complete"])
        self.assertEqual(report["results"][0]["error"], "RuntimeError")
        self.assertNotIn("secret", json.dumps(report))

    def test_invalid_response_does_not_abort_other_cases(self):
        for bad in [None, {"output":"ok", "input_tokens":True,"output_tokens":1}, {"output":"ok", "input_tokens":-1,"output_tokens":1}, {"output":12,"input_tokens":1,"output_tokens":1}]:
            report = self.lab.report(self.lab.run(self.prompt["id"], CASES, lambda p,c: bad if c == "a" else self.adapter(p,c)))
            self.assertEqual(report["summary"]["errors"], 1)
            self.assertEqual(report["summary"]["successes"], 1)

    def test_json_rubric_rejects_boolean_number_and_invalid_json(self):
        for output in ['{"count":true}', '{"count":NaN}', '[1]', 'not JSON', '{}']:
            report = self.lab.report(self.lab.run(self.prompt["id"], [CASES[1]], lambda p,c:{"output":output,"input_tokens":0,"output_tokens":0}))
            self.assertEqual(report["summary"]["mean_score"], 0)

    def test_case_validation_and_rates(self):
        invalid = [[], [CASES[0], CASES[0]], [{**CASES[0],"checks":{}}], [{**CASES[0],"checks":{"contains":[]}}], [{**CASES[0],"checks":{"magic":True}}]]
        for cases in invalid:
            with self.assertRaises(ValueError):
                self.lab.run(self.prompt["id"], cases, self.adapter)
        for price in [True, -1, "NaN", "Infinity", "bad", "0.0000000000001"]:
            with self.assertRaises(ValueError):
                self.run_fixture(input_per_1000=price)

    def test_comparison_scores_and_mismatched_fixtures(self):
        first = self.run_fixture()
        second = self.lab.report(self.lab.run(self.prompt["id"], CASES, lambda p,c:{"output":"wrong","input_tokens":100,"output_tokens":50}))
        # "wrong" contains "o", passing one of case A's two checks.
        self.assertEqual(second["results"][0]["score"], .5)
        self.assertEqual(compare(first,second)["score_delta"], -.75)
        self.assertTrue(compare(first,second)["cost_comparable"])
        changed = copy.deepcopy(CASES)
        changed[0]["checks"]["exact"] = "different"
        other = self.lab.report(self.lab.run(self.prompt["id"], changed, self.adapter))
        with self.assertRaises(ValueError):
            compare(first, other)

    def test_comparison_cost_requires_matching_rates_and_currency(self):
        a, b = self.run_fixture(), self.run_fixture(currency="NGN")
        self.assertIsNone(compare(a,b)["cost_delta"])
        c = self.run_fixture(input_per_1000="1")
        self.assertFalse(compare(a,c)["cost_comparable"])

    def test_interrupted_run_retains_completed_case_and_pending_count(self):
        def interrupted(prompt, case):
            if case == "b":
                raise SystemExit("simulated process interruption")
            return self.adapter(prompt, case)
        with self.assertRaises(SystemExit):
            self.lab.run(self.prompt["id"], CASES, interrupted)
        report = self.lab.report(1)
        self.assertEqual(report["state"], "running")
        self.assertEqual(report["summary"]["recorded"], 1)
        self.assertEqual(report["summary"]["pending"], 1)
        with self.assertRaises(ValueError):
            compare(report, report)

    def test_restart_preserves_snapshot_and_results(self):
        report = self.run_fixture()
        other = Lab(self.path)
        self.assertEqual(other.report(report["id"]), report)

    def test_html_escapes_untrusted_model_text(self):
        run = self.lab.run(self.prompt["id"], [CASES[0]], lambda p,c:{"output":"<script>alert(1)</script>","input_tokens":1,"output_tokens":1})
        html = self.lab.html_report(run)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('scope="col"', html)

    def test_empty_success_denominator_and_bad_clock(self):
        report = self.lab.report(self.lab.run(self.prompt["id"], CASES, lambda p,c:None))
        self.assertIsNone(report["summary"]["mean_success_seconds"])
        self.assertEqual(report["summary"]["mean_score"], 0)
        ticks = iter([2,1,4,3])
        self.lab.timer = lambda: next(ticks)
        self.assertEqual(self.run_fixture()["summary"]["errors"], 2)

    def test_missing_prompt_and_tampered_template(self):
        with self.assertRaises(KeyError):
            self.lab.run(999, CASES, self.adapter)
        with self.assertRaises(ValueError):
            self.lab.add_prompt("invalid", "No marker")
        with self.lab.db(True) as db:
            db.execute("UPDATE prompts SET template='changed' WHERE id=?", (self.prompt["id"],))
        with self.assertRaises(ValueError):
            self.run_fixture()


if __name__ == "__main__":
    unittest.main()
