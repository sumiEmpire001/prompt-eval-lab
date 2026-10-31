# Prompt Eval Lab

Versioned prompts, repeatable fixture runs and rubric/cost/latency comparisons. Python 3.10+; SQLite and standard library only. The included adapter returns labelled synthetic fixture responses without calling any external model.

```sh
python -m unittest discover -s tests -v
python demo.py
python demo.py --html report.html
```

Open the generated report.html in a browser. Model output is escaped and the report is self-contained. The demo compares two explicit prompt versions on the same fixture; its improvement is manufactured fixture behavior, not a measured improvement from a real model.

```python
from eval_lab import Lab, compare
lab = Lab('evaluations.db')
prompt = lab.add_prompt('demo', 'Respond to: {{input}}')
cases = [{'id':'hello', 'input':'Say hello', 'checks':{'exact':'Hello'}}]
def local_adapter(rendered_prompt, case_id):
    return {'output':'Hello', 'input_tokens':10, 'output_tokens':2}
run = lab.run(prompt['id'], cases, local_adapter, provider='local-fixture', input_per_1000='0', output_per_1000='0')
print(lab.report(run))
```

Prompt registration creates immutable numbered versions. Each run snapshots the exact prompt, cases, rubrics, provider label and supplied prices. Templates substitute exactly one literal {{input}} marker; no expression evaluation occurs. Prompt integrity is checked before use. Each case result commits independently; an interrupted run retains completed results and reports remaining cases as pending. Interrupted runs cannot be resumed through this API; a new run may repeat provider calls and charges.

Rubrics support exact text, all specified case-sensitive snippets, and exact top-level JSON field values. JSON fields distinguish booleans from numbers and reject nonfinite constants. Checks within a case have equal weight; every case has equal weight. Failed calls score zero. The overall mean uses all fixture cases as denominator, including pending cases in interrupted runs. These are explicit automated rubrics, not a semantic-quality or safety certification.

Adapters return exactly output, input_tokens and output_tokens. Failures are isolated so other cases continue. Error messages are discarded; only exception type is retained. Token counts must be nonnegative integers. Costs use Decimal and rates supplied per1000 tokens, with at most12 decimal places; they are estimates, not invoices or verified model pricing. Costs of failed/invalid responses remain unknown, and cost_complete stays false. Reports separate known cost from that missing accounting.

compare(left,right) requires completed runs with identical fixture hashes. It reports score/error changes and mean successful-call latency changes. Cost deltas are available only with complete token accounting and identical rates/currency. Latency averages may cover different successful subsets and are not a paired statistical benchmark. Live provider reproducibility, timeouts, retries, rate limits and spend caps belong in a separately configured adapter; none is included.

14 local tests cover concurrent prompt versions, repeatable fixtures, partial failures, exact costs, JSON rubrics, mismatched comparisons, invalid adapters, interrupted runs, restart, HTML escaping and integrity checks. GitHub Actions runs tests and the synthetic demo on Python3.10/3.12/3.13. Actions → Tests and demo → Run workflow repeats them.

Limits: trusted local tool, no authentication/encryption, no UI server, no automatic provider installation and no live provider verification. Prompts, inputs and successful outputs are stored in plaintext. Do not use confidential inputs in shared reports. Up to1000 cases, 2MiB fixture data,10000-character inputs,100000-character outputs. Hashes detect accidental drift, not a database administrator rewriting both content and evidence. A malicious adapter is arbitrary Python code; only use trusted adapters.
