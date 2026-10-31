import hashlib
import html
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, localcontext


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def label(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("Use a nonempty label up to 100 characters")
    return value.strip()


def rate(value):
    if type(value) not in (int, str, Decimal):
        raise ValueError("Rates must be integer, decimal string or Decimal")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid rate") from exc
    if not result.is_finite() or result < 0 or result > 1000000 or result.as_tuple().exponent < -12:
        raise ValueError("Use finite rates 0..1000000 with at most 12 decimal places")
    return result


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ValueError("Use 1..1000 cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "input", "checks"}:
            raise ValueError("Each case needs exactly id, input and checks")
        ident = label(case["id"])
        if ident != case["id"] or ident in seen:
            raise ValueError("Case IDs must be trimmed and unique")
        seen.add(ident)
        if not isinstance(case["input"], str) or len(case["input"]) > 10000:
            raise ValueError("Case input must be text up to 10000 characters")
        checks = case["checks"]
        if not isinstance(checks, dict) or not checks or set(checks) - {"exact", "contains", "json_fields"}:
            raise ValueError("Use exact, contains and/or json_fields checks")
        if "exact" in checks and not isinstance(checks["exact"], str):
            raise ValueError("Exact check must be text")
        if "contains" in checks and (not isinstance(checks["contains"], list) or not checks["contains"] or any(not isinstance(v, str) or not v for v in checks["contains"])):
            raise ValueError("Contains needs nonempty text snippets")
        if "json_fields" in checks and (not isinstance(checks["json_fields"], dict) or not checks["json_fields"] or any(not isinstance(key, str) for key in checks["json_fields"])):
            raise ValueError("JSON fields must be a nonempty object")
    if len(encode(cases).encode()) > 2*1024*1024:
        raise ValueError("Evaluation fixture exceeds 2 MiB")


def score(output, checks):
    results = {}
    if "exact" in checks:
        results["exact"] = output == checks["exact"]
    if "contains" in checks:
        results["contains"] = all(snippet in output for snippet in checks["contains"])
    if "json_fields" in checks:
        try:
            def invalid(value):
                raise ValueError("Nonfinite JSON constant")
            parsed = json.loads(output, parse_constant=invalid)
            results["json_fields"] = isinstance(parsed, dict) and all(key in parsed and encode(parsed[key]) == encode(value) for key, value in checks["json_fields"].items())
        except (ValueError, TypeError, RecursionError):
            results["json_fields"] = False
    return results


class Lab:
    def __init__(self, path, timer=time.perf_counter):
        self.path, self.timer = str(path), timer
        with self.db() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS prompts(
                id INTEGER PRIMARY KEY,name TEXT NOT NULL,version INTEGER NOT NULL,template TEXT NOT NULL,
                digest TEXT NOT NULL,UNIQUE(name,version));
                CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY,prompt INTEGER NOT NULL REFERENCES prompts(id),snapshot TEXT NOT NULL,state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS results(
                run INTEGER NOT NULL REFERENCES runs(id),case_id TEXT NOT NULL,result TEXT NOT NULL,PRIMARY KEY(run,case_id));""")

    @contextmanager
    def db(self, write=False):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def add_prompt(self, name, template):
        name = label(name)
        if not isinstance(template, str) or template.count("{{input}}") != 1 or len(template) > 20000:
            raise ValueError("Template needs exactly one {{input}} marker and at most 20000 characters")
        with self.db(True) as db:
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM prompts WHERE name=?", (name,)).fetchone()[0]
            ident = db.execute("INSERT INTO prompts(name,version,template,digest) VALUES(?,?,?,?)", (name, version, template, fingerprint(template))).lastrowid
            return {"id": ident, "name": name, "version": version, "digest": fingerprint(template)}

    def run(self, prompt, cases, adapter, *, provider="local-fixture", input_per_1000="0", output_per_1000="0", currency="USD"):
        validate_cases(cases)
        provider, currency = label(provider), label(currency)
        incoming, outgoing = rate(input_per_1000), rate(output_per_1000)
        # Snapshot before invoking the adapter so caller mutations cannot alter the fixture.
        cases = json.loads(encode(cases))
        with self.db(True) as db:
            row = db.execute("SELECT * FROM prompts WHERE id=?", (prompt,)).fetchone()
            if row is None:
                raise KeyError(prompt)
            if fingerprint(row["template"]) != row["digest"]:
                raise ValueError("Prompt integrity mismatch")
            snapshot = {"prompt": dict(row), "cases": cases, "fixture_hash": fingerprint(cases), "provider": provider,
                        "input_per_1000": str(incoming), "output_per_1000": str(outgoing), "currency": currency}
            ident = db.execute("INSERT INTO runs(prompt,snapshot,state) VALUES(?,?,'running')", (prompt, encode(snapshot))).lastrowid
        failures = 0
        for case in cases:
            started = self.timer()
            try:
                response = adapter(row["template"].replace("{{input}}", case["input"]), case["id"])
                elapsed = self.timer() - started
                if not isinstance(response, dict) or set(response) != {"output", "input_tokens", "output_tokens"}:
                    raise ValueError("Adapter response shape invalid")
                if not isinstance(response["output"], str) or len(response["output"]) > 100000 or any(type(response[key]) is not int or not 0 <= response[key] <= 1000000000 for key in ("input_tokens", "output_tokens")):
                    raise ValueError("Adapter output or token counts invalid")
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid monotonic timer")
                checks = score(response["output"], case["checks"])
                with localcontext() as ctx:
                    ctx.prec = 50
                    cost = (incoming*response["input_tokens"] + outgoing*response["output_tokens"])/1000
                result = {"id": case["id"], "status": "ok", **response, "checks": checks,
                          "score": sum(checks.values())/len(checks), "seconds": elapsed, "cost": str(cost), "error": None}
            except Exception as exc:
                failures += 1
                # Provider exception messages can contain secrets; persist type only.
                result = {"id": case["id"], "status": "error", "score": 0, "checks": {}, "output": None,
                          "input_tokens": None, "output_tokens": None, "seconds": None, "cost": None, "error": type(exc).__name__}
            with self.db(True) as db:
                db.execute("INSERT INTO results VALUES(?,?,?)", (ident, case["id"], encode(result)))
        with self.db(True) as db:
            db.execute("UPDATE runs SET state=? WHERE id=?", ("completed_with_errors" if failures else "completed", ident))
        return ident

    def report(self, run):
        with self.db() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
            if row is None:
                raise KeyError(run)
            snapshot = json.loads(row["snapshot"])
            by_id = {r["case_id"]: json.loads(r["result"]) for r in db.execute("SELECT * FROM results WHERE run=?", (run,))}
            results = [by_id[case["id"]] for case in snapshot["cases"] if case["id"] in by_id]
        successful = [r for r in results if r["status"] == "ok"]
        count = len(snapshot["cases"])
        with localcontext() as ctx:
            ctx.prec = 50
            total_cost = str(sum((Decimal(r["cost"]) for r in successful), Decimal(0)))
        return {"id": run, "state": row["state"], "snapshot": snapshot, "results": results,
                "summary": {"cases": count, "recorded": len(results), "successes": len(successful),
                            "errors": len(results)-len(successful), "pending": count-len(results),
                            "mean_score": sum(r["score"] for r in results)/count,
                            "mean_success_seconds": sum(r["seconds"] for r in successful)/len(successful) if successful else None,
                            "known_cost": total_cost,
                            "cost_complete": len(successful) == count}}

    def html_report(self, run):
        report = self.report(run)
        esc = lambda value: html.escape(str(value))
        rows = "".join("<tr>"+"".join("<td>"+esc(result[key])+"</td>" for key in ("id", "status", "score", "seconds", "cost", "output", "error"))+"</tr>" for result in report["results"])
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Prompt evaluation report</title><style>body{font:1rem system-ui;max-width:75rem;margin:2rem auto;padding:1rem;color:#182b28;background:#f7f8f4}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.7rem;border:1px solid #a4b3ad;overflow-wrap:anywhere}main{overflow:auto}pre{white-space:pre-wrap}</style><main><h1>Prompt evaluation report</h1><p>Local evaluation evidence. Costs are estimates from supplied rates and reported tokens.</p><pre>'+esc(encode(report["summary"]))+'</pre><table><caption>Run '+str(run)+' — '+esc(report["state"])+'</caption><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in ("Case", "Status", "Score", "Seconds", "Cost", "Output", "Error"))+'</tr></thead><tbody>'+rows+'</tbody></table></main></html>'


def compare(left, right):
    if left["state"] not in ("completed", "completed_with_errors") or right["state"] not in ("completed", "completed_with_errors"):
        raise ValueError("Compare completed runs only")
    if left["snapshot"]["fixture_hash"] != right["snapshot"]["fixture_hash"]:
        raise ValueError("Comparison requires exactly matching cases and rubrics")
    a, b = left["summary"], right["summary"]
    comparable_cost = a["cost_complete"] and b["cost_complete"] and all(left["snapshot"][key] == right["snapshot"][key] for key in ("currency", "input_per_1000", "output_per_1000"))
    with localcontext() as ctx:
        ctx.prec = 50
        cost_delta = str(Decimal(b["known_cost"])-Decimal(a["known_cost"])) if comparable_cost else None
    return {"score_delta": b["mean_score"]-a["mean_score"], "error_delta": b["errors"]-a["errors"],
            "cost_delta": cost_delta,
            "cost_comparable": comparable_cost,
            "latency_delta": b["mean_success_seconds"]-a["mean_success_seconds"] if a["mean_success_seconds"] is not None and b["mean_success_seconds"] is not None else None}
