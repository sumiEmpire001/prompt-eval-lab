import argparse
import json
import tempfile
from pathlib import Path
from eval_lab import Lab, compare

parser = argparse.ArgumentParser(description="Provider-free synthetic prompt evaluation")
parser.add_argument("--html", type=Path, help="Write an offline HTML report to this explicit path")
args = parser.parse_args()
cases = [{"id":"greeting", "input":"Say hello", "checks":{"exact":"Hello", "contains":["Hello"]}},
         {"id":"structured", "input":"Return status JSON", "checks":{"json_fields":{"status":"ok"}}}]
with tempfile.TemporaryDirectory() as directory:
    lab = Lab(Path(directory) / "lab.db")
    first = lab.add_prompt("synthetic-baseline", "Baseline: {{input}}")
    second = lab.add_prompt("synthetic-baseline", "Revised: {{input}}")
    def fixture(prompt, case):
        output = "Hello" if case == "greeting" else '{"status":"ok"}' if prompt.startswith("Revised:") else "ok"
        return {"output":output, "input_tokens":10, "output_tokens":4}
    one = lab.run(first["id"], cases, fixture)
    two = lab.run(second["id"], cases, fixture)
    print("Synthetic fixture only — no external model called")
    print(json.dumps(lab.report(two)["summary"], indent=2))
    print(json.dumps(compare(lab.report(one), lab.report(two)), indent=2))
    if args.html:
        args.html.write_text(lab.html_report(two), encoding="utf-8")
        print("HTML report:", args.html)
