"""CPU-only checks for the deterministic MiMo validation suite generator+scorer."""
import ast
import importlib.util
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("mimo_validation_suite_under_test",
                                               SCRIPTS / "mimo_validation_suite.py")
suite_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite_mod)


def _by_family(suite):
    grouped = {}
    for task in suite["tasks"]:
        grouped.setdefault(task["family"], []).append(task)
    return grouped


class SuiteShapeTests(unittest.TestCase):
    def test_nonfinite_json_number_rejected(self):
        self.assertIsNotNone(suite_mod.extract_json('1e999')[1])

    def test_duplicate_json_key_rejected(self):
        self.assertIsNotNone(suite_mod.extract_json('{"x": 1, "x": 2}')[1])

    def test_counts_categories_families(self):
        suite = suite_mod.build_suite()
        self.assertEqual(suite["seed"], 20260922)
        self.assertEqual(suite["system"],
                         "Follow the user instructions. Return only the requested JSON value.")
        self.assertEqual(suite["quality_max_tokens"], 128)
        self.assertEqual(len(suite["tasks"]), 64)
        cats = Counter(t["category"] for t in suite["tasks"])
        self.assertEqual(set(cats), {"code", "reasoning", "tools", "general"})
        for cat, count in cats.items():
            self.assertEqual(count, 16, msg=cat)
        fams = Counter(t["family"] for t in suite["tasks"])
        self.assertEqual(len(fams), 16)
        for fam, count in fams.items():
            self.assertEqual(count, 4, msg=fam)
        ids = [t["id"] for t in suite["tasks"]]
        self.assertEqual(len(set(ids)), 64)
        # tools split: 8 choose-a-tool + 8 normalize/extract
        by_fam = _by_family(suite)
        self.assertEqual(len(by_fam["tools_choose_single"]) + len(by_fam["tools_choose_multi"]), 8)
        self.assertEqual(len(by_fam["tools_normalize"]) + len(by_fam["tools_extract"]), 8)

    def test_bounded_prompts_and_expected(self):
        suite = suite_mod.build_suite()
        for task in suite["tasks"]:
            self.assertTrue(task["prompt"], msg=task["id"])
            self.assertLessEqual(len(task["prompt"]), 2000, msg=task["id"])
            blob = json.dumps(task["expected"], ensure_ascii=False)
            self.assertLessEqual(len(blob), 600, msg=task["id"])

    def test_deterministic(self):
        first = suite_mod.build_suite()
        second = suite_mod.build_suite()
        self.assertEqual(first, second)


class CodeFamilyTests(unittest.TestCase):
    def test_filter_reduce(self):
        for task in _by_family(suite_mod.build_suite())["code_filter_reduce"]:
            values = ast.literal_eval(re.search(r"values = (\[.*?\])", task["prompt"]).group(1))
            d = int(re.search(r"d = (\d+)", task["prompt"]).group(1))
            r = int(re.search(r"r = (\d+)", task["prompt"]).group(1))
            body = task["prompt"]
            if "total += x * x" in body:
                ref = sum(x * x for x in values if x % d == r)
            elif "total += 2 * x + 1" in body:
                ref = sum(2 * x + 1 for x in values if x % d != r)
            elif "result = total * count" in body:
                sel = [x for x in values if x % d == r]
                ref = sum(sel) * len(sel)
            else:
                ref = sum(x for x in values if x % d == r)
            self.assertEqual(task["expected"], ref, msg=task["id"])

    def test_group_count(self):
        for task in _by_family(suite_mod.build_suite())["code_group_count"]:
            items = ast.literal_eval(re.search(r"items = (\[.*?\])", task["prompt"]).group(1))
            seen = []
            for item in items:
                if item not in seen:
                    seen.append(item)
            ref = [[key, items.count(key)] for key in seen]
            self.assertEqual(task["expected"], ref, msg=task["id"])

    def test_loop_trace(self):
        for task in _by_family(suite_mod.build_suite())["code_loop_trace"]:
            a, b = map(int, re.search(r"for i in range\((\d+), (\d+)\)", task["prompt"]).groups())
            c, d = map(int, re.search(r"for j in range\((\d+), (\d+)\)", task["prompt"]).groups())
            m, r = map(int, re.search(r"% (\d+) == (\d+)", task["prompt"]).groups())
            uses_product = "(i * j)" in task["prompt"]
            if "total += i + j" in task["prompt"]:
                ref = sum(i + j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
            elif "total += i * j" in task["prompt"]:
                ref = sum(i * j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
            elif "total += i - j" in task["prompt"]:
                ref = sum(i - j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
            else:
                self.assertTrue(uses_product, msg=task["id"])
                ref = sum(1 for i in range(a, b) for j in range(c, d) if (i * j) % m == r)
            self.assertEqual(task["expected"], ref, msg=task["id"])

    def test_dict_state(self):
        pat = re.compile(r'state\["([^"]+)"\] (\+=|=) (.+)')
        for task in _by_family(suite_mod.build_suite())["code_dict_state"]:
            init = json.loads(re.search(r"state = (\{.*?\})", task["prompt"]).group(1))
            cur = dict(init)
            for line in task["prompt"].splitlines():
                m = pat.match(line.strip())
                if not m:
                    continue
                key, op, expr = m.groups()
                if op == "+=":
                    cur[key] += int(expr)
                elif expr.endswith(" * 2"):
                    src = re.search(r'state\["([^"]+)"\]', expr).group(1)
                    cur[key] = cur[src] * 2
                elif "+" in expr and "state" in expr:
                    parts = re.findall(r'state\["([^"]+)"\]', expr)
                    num = re.search(r"\+\s*(\d+)", expr)
                    if num and len(parts) == 1:
                        cur[key] = cur[parts[0]] + int(num.group(1))
                    else:
                        cur[key] = cur[parts[0]] + cur[parts[1]]
                else:
                    self.fail(f"unparsed op line: {line}")
            self.assertEqual(task["expected"], cur, msg=task["id"])


class ReasoningFamilyTests(unittest.TestCase):
    def test_inventory(self):
        for task in _by_family(suite_mod.build_suite())["reasoning_inventory"]:
            start = int(re.search(r"with (\d+)", task["prompt"]).group(1))
            cur = start
            for verb, amt in re.findall(r"(Received|Shipped|Discarded as damaged|Accepted as returns) (\d+)",
                                        task["prompt"]):
                amt = int(amt)
                cur += amt if verb in ("Received", "Accepted as returns") else -amt
            self.assertEqual(task["expected"], cur, msg=task["id"])

    def test_ratio(self):
        for task in _by_family(suite_mod.build_suite())["reasoning_ratio"]:
            prompt = task["prompt"]
            if "crates" in prompt:
                crates, boxes, parts = map(int, re.search(
                    r"(\d+) crates.*holds (\d+) boxes.*holds (\d+) parts", prompt).groups())
                removed = int(re.search(r"(\d+) parts\nare set aside|(\d+) parts are set aside",
                                        prompt).group(0).split()[0])
                self.assertEqual(task["expected"], crates * boxes * parts - removed,
                                 msg=task["id"])
            elif "alloy" in prompt:
                a, b = map(int, re.search(r"ratio (\d+):(\d+)", prompt).groups())
                total = int(re.search(r"uses (\d+) kg", prompt).group(1))
                units = total // (a + b)
                self.assertEqual(total % (a + b), 0, msg=task["id"])
                self.assertEqual(task["expected"], {"a": a * units, "b": b * units},
                                 msg=task["id"])
            elif "recipe" in prompt:
                flour, serves = map(int, re.search(r"uses (\d+) grams.*?make (\d+) servings",
                                                  prompt).groups())
                need = int(re.search(r"make (\d+) servings at", prompt).group(1))
                self.assertEqual(need % serves, 0, msg=task["id"])
                self.assertEqual(task["expected"], flour * (need // serves), msg=task["id"])
            else:
                per_hour, hours, days = map(int, re.search(
                    r"makes (\d+) items per hour.*runs (\d+) hours per day\nfor (\d+) days|"
                    r"makes (\d+) items per hour.*runs (\d+) hours per day for (\d+) days",
                    prompt.replace("\n", " ")).groups()[-3:])
                defective = int(re.search(r"(\d+) items fail", prompt).group(1))
                self.assertEqual(task["expected"], per_hour * hours * days - defective,
                                 msg=task["id"])

    def test_order_constraints_hold_and_unique(self):
        for task in _by_family(suite_mod.build_suite())["reasoning_order"]:
            expected = task["expected"]
            self.assertEqual(len(expected), len(set(expected)), msg=task["id"])
            edges = re.findall(r"(\w+) came before (\w+)\.", task["prompt"])
            self.assertTrue(edges, msg=task["id"])
            pos = {name: i for i, name in enumerate(expected)}
            for before, after in edges:
                self.assertLess(pos[before], pos[after], msg=(task["id"], before, after))
            for name in re.findall(r"(\w+) was not last\.", task["prompt"]):
                self.assertNotEqual(expected[-1], name, msg=task["id"])
            for name in re.findall(r"(\w+) was not first\.", task["prompt"]):
                self.assertNotEqual(expected[0], name, msg=task["id"])
            # Independent topological reconstruction must match expected exactly.
            nodes = set(pos)
            incoming = {n: set() for n in nodes}
            outgoing = {n: set() for n in nodes}
            for before, after in edges:
                outgoing[before].add(after)
                incoming[after].add(before)
            order = []
            while incoming:
                ready = sorted(n for n, preds in incoming.items() if not preds)
                self.assertEqual(len(ready), 1, msg=task["id"])
                node = ready[0]
                order.append(node)
                del incoming[node]
                for succ in outgoing[node]:
                    if succ in incoming:
                        incoming[succ].discard(node)
            self.assertEqual(order, expected, msg=task["id"])

    def test_schedule(self):
        for task in _by_family(suite_mod.build_suite())["reasoning_schedule"]:
            durations = {}
            prereqs = {}
            for name, dur, pre in re.findall(
                    r"- Task (\w): takes (\d+) hours; prerequisites: ([^.]*)\.", task["prompt"]):
                durations[name] = int(dur)
                prereqs[name] = [] if pre.strip() == "none" else [p.strip() for p in pre.split(",")]
            finish = {}
            for name in ("A", "B", "C", "D"):
                start = max((finish[p] for p in prereqs[name]), default=0)
                finish[name] = start + durations[name]
            self.assertEqual(task["expected"], max(finish.values()), msg=task["id"])


class ToolsFamilyTests(unittest.TestCase):
    def _offered_and_request(self, task):
        head, request = task["prompt"].split("\nRequest: ", 1)
        offered = json.loads(head.split("\n", 1)[1])
        return offered, request

    def test_choose_single(self):
        for task in _by_family(suite_mod.build_suite())["tools_choose_single"]:
            offered, request = self._offered_and_request(task)
            names = [t["name"] for t in offered]
            self.assertIn(task["expected"]["name"], names, msg=task["id"])
            args = task["expected"]["arguments"]
            self.assertTrue(args, msg=task["id"])
            for value in args.values():
                self.assertIn(str(value), request, msg=(task["id"], value))

    def test_choose_multi(self):
        for task in _by_family(suite_mod.build_suite())["tools_choose_multi"]:
            offered, request = self._offered_and_request(task)
            self.assertIn(task["expected"]["name"], [t["name"] for t in offered],
                          msg=task["id"])
            blob = json.dumps(task["expected"]["arguments"], ensure_ascii=False)
            self.assertTrue(len(blob) > 10, msg=task["id"])
            if task["expected"]["name"] == "route_plan":
                self.assertIsInstance(task["expected"]["arguments"]["options"]["avoid_tolls"], bool)
            if task["expected"]["name"] == "survey_create":
                self.assertIsInstance(task["expected"]["arguments"]["anonymous"], bool)

    def test_normalize(self):
        for task in _by_family(suite_mod.build_suite())["tools_normalize"]:
            raw = json.loads(task["prompt"].split("Normalize this JSON value:\n", 1)[1].split("\nRules:", 1)[0])
            expected = task["expected"]
            if "ID" in raw:
                self.assertEqual(expected, {"id": int(raw["ID"]), "name": raw["Name"].strip(),
                                             "tags": sorted(raw["tags"])}, msg=task["id"])
            elif "Sensor" in raw:
                self.assertEqual(expected, {"sensor": raw["Sensor"],
                                             "reading": int(raw[" Reading "]),
                                             "unit": raw["unit"].strip()}, msg=task["id"])
            elif "user_id" in raw:
                self.assertEqual(expected, {"user_id": int(raw["user_id"]),
                                             "email": raw["email"].strip(),
                                             "roles": sorted(raw["roles"])}, msg=task["id"])
            else:
                self.assertEqual(expected, {"order": int(raw["order"]), "qty": int(raw["qty"]),
                                             "priority": raw["priority"].strip()}, msg=task["id"])

    def test_extract(self):
        for task in _by_family(suite_mod.build_suite())["tools_extract"]:
            record = task["prompt"].split("Record:\n", 1)[1].split("\n", 1)[0]
            expected = task["expected"]
            if record.startswith("Order #"):
                m = re.match(r"Order #(\d+) \| customer: (\w+) \| items: (\d+)x (\w+), "
                             r"(\d+)x (\w+) \| priority: (\w+)", record)
                self.assertIsNotNone(m, msg=task["id"])
                oid, name, q1, i1, q2, i2, prio = m.groups()
                self.assertEqual(expected, {"order": int(oid), "customer": name,
                                             "items": [{"name": i1, "qty": int(q1)},
                                                       {"name": i2, "qty": int(q2)}],
                                             "priority": prio}, msg=task["id"])
            elif record.startswith("[2026"):
                m = re.match(r"\[(.*?) .*?\] (\w+) (\w+): (\w+) retries=(\d+) latency_ms=(\d+)",
                             record)
                self.assertIsNotNone(m, msg=task["id"])
                date, level, service, code, retries, lat = m.groups()
                self.assertEqual(expected, {"date": date, "level": level, "service": service,
                                             "code": code, "retries": int(retries),
                                             "latency_ms": int(lat)}, msg=task["id"])
            elif record.startswith("SKU-"):
                m = re.match(r"(\S+) count=(\d+) bins=(\w+),(\w+) flag=(\w+)", record)
                self.assertIsNotNone(m, msg=task["id"])
                sku, count, b1, b2, flag = m.groups()
                self.assertEqual(expected, {"sku": sku, "count": int(count),
                                             "bins": [b1, b2], "flag": flag}, msg=task["id"])
            else:
                m = re.match(r"Ticket (\d+) \[(\w+)\] (\w+) assignee=(\w+) hours=(\d+)", record)
                self.assertIsNotNone(m, msg=task["id"])
                tid, sev, system, person, hours = m.groups()
                self.assertEqual(expected, {"ticket": int(tid), "severity": sev,
                                             "system": system, "assignee": person,
                                             "hours": int(hours)}, msg=task["id"])


class GeneralFamilyTests(unittest.TestCase):
    def test_lookup(self):
        for task in _by_family(suite_mod.build_suite())["general_lookup"]:
            prompt = task["prompt"]
            if prompt.startswith("Station roster"):
                rows = dict(re.findall(r"(S-\d) \| city: (\w+)", prompt))
                asked = re.search(r"station (S-\d)", prompt).group(1)
                self.assertEqual(task["expected"], rows[asked], msg=task["id"])
            elif prompt.startswith("Crew roster"):
                rows = dict(re.findall(r"(\w+) \| role: (\w+)", prompt))
                asked = re.search(r"role of (\w+)", prompt).group(1)
                self.assertEqual(task["expected"], rows[asked], msg=task["id"])
            elif prompt.startswith("Stock table"):
                rows = {sku: int(stock) for sku, stock in re.findall(r"(SKU-\d+) \| stock: (\d+)", prompt)}
                asked = re.search(r"stock of (SKU-\d+)", prompt).group(1)
                self.assertEqual(task["expected"], rows[asked], msg=task["id"])
            else:
                rows = dict(re.findall(r"(Log \d+) \| shelf: (H-\d)", prompt))
                asked = re.search(r"holds (Log \d+)", prompt).group(1)
                self.assertEqual(task["expected"], rows[asked], msg=task["id"])

    def test_sort(self):
        for task in _by_family(suite_mod.build_suite())["general_sort"]:
            prompt = task["prompt"]
            if "status active" in prompt:
                rows = re.findall(r"(S-\d) \| status: (\w+)", prompt)
                ref = sorted(code for code, status in rows if status == "active")
            elif "lowest to highest stock" in prompt:
                rows = [(sku, int(stock)) for sku, stock in
                        re.findall(r"(SKU-\d+) \| stock: (\d+)", prompt)]
                ref = [sku for sku, _ in sorted(rows, key=lambda p: p[1])]
            elif "oldest to youngest" in prompt:
                rows = [(name, int(age)) for name, age in
                        re.findall(r"(\w+) \| age: (\d+)", prompt)]
                ref = [name for name, _ in sorted(rows, key=lambda p: -p[1])]
            else:
                rows = re.findall(r"(Log \d+) \| shelf: (H-\d)", prompt)
                ref = sorted(title for title, shelf in rows if shelf == "H-1")
            self.assertEqual(task["expected"], ref, msg=task["id"])

    def test_filter(self):
        for task in _by_family(suite_mod.build_suite())["general_filter"]:
            prompt = task["prompt"]
            if prompt.startswith("Gauge readings"):
                rows = [(gid, int(val)) for gid, val in re.findall(r"(G-\d+) \| reading: (\d+)", prompt)]
                thresh = int(re.search(r"at least (\d+)", prompt).group(1))
                ref = [gid for gid, val in rows if val >= thresh]
            elif prompt.startswith("Crew shifts"):
                rows = re.findall(r"(\w+) \| shift: (\w+)", prompt)
                ref = [name for name, shift in rows if shift == "night"]
            elif prompt.startswith("Stock table"):
                rows = [(sku, int(val)) for sku, val in re.findall(r"(SKU-\d+) \| stock: (\d+)", prompt)]
                thresh = int(re.search(r"below (\d+)", prompt).group(1))
                ref = [sku for sku, val in rows if val < thresh]
            else:
                rows = [(title, int(pages)) for title, pages in
                        re.findall(r"(Log \d+) \| pages: (\d+)", prompt)]
                thresh = int(re.search(r"more than (\d+) pages", prompt).group(1))
                ref = [title for title, pages in rows if pages > thresh]
            self.assertEqual(task["expected"], ref, msg=task["id"])
            self.assertTrue(0 < len(ref) < 6, msg=task["id"])

    def test_aggregate(self):
        for task in _by_family(suite_mod.build_suite())["general_aggregate"]:
            prompt = task["prompt"]
            if prompt.startswith("Stock table"):
                rows = re.findall(r"SKU-\d+ \| stock: (\d+) \| status: (\w+)", prompt)
                sel = [int(stock) for stock, status in rows if status == "active"]
                self.assertEqual(task["expected"], {"count": len(sel), "total": sum(sel)},
                                 msg=task["id"])
            elif prompt.startswith("Gauge readings"):
                readings = list(map(int, re.findall(r"reading: (\d+)", prompt)))
                self.assertEqual(task["expected"], {"min": min(readings), "max": max(readings)},
                                 msg=task["id"])
            elif prompt.startswith("Crew hours"):
                rows = re.findall(r"\w+ \| hours: (\d+) \| shift: (\w+)", prompt)
                sel = [int(hours) for hours, shift in rows if shift == "night"]
                self.assertEqual(task["expected"], {"count": len(sel), "total_hours": sum(sel)},
                                 msg=task["id"])
            else:
                rows = re.findall(r"Log \d+ \| pages: (\d+) \| shelf: (H-\d)", prompt)
                sel = [int(pages) for pages, shelf in rows if shelf == "H-1"]
                self.assertEqual(task["expected"], {"count": len(sel), "pages": sum(sel)},
                                 msg=task["id"])


class ScorerTests(unittest.TestCase):
    def test_bool_is_not_number_and_integral_float_ok(self):
        self.assertFalse(suite_mod.json_equal(1, True))
        self.assertFalse(suite_mod.json_equal(True, 1))
        self.assertFalse(suite_mod.json_equal(0, False))
        self.assertTrue(suite_mod.json_equal(1, 1.0))
        self.assertTrue(suite_mod.json_equal(1.0, 1))
        self.assertFalse(suite_mod.json_equal(1, 1.5))
        self.assertTrue(suite_mod.json_equal({"a": 1, "b": 2}, {"b": 2.0, "a": 1}))
        self.assertFalse(suite_mod.json_equal([1, 2], [2, 1]))

    def test_fenced_and_malformed(self):
        value, error = suite_mod.extract_json('```json\n{"a": 1}\n```')
        self.assertIsNone(error)
        self.assertEqual(value, {"a": 1})
        value, error = suite_mod.extract_json('```\n[1, 2]\n```')
        self.assertIsNone(error)
        self.assertEqual(value, [1, 2])
        _, error = suite_mod.extract_json('Here is {"a": 1}')
        self.assertIsNotNone(error)
        _, error = suite_mod.extract_json('{"a": 1} trailing')
        self.assertIsNotNone(error)
        _, error = suite_mod.extract_json('```json\n{"a": 1}\n```\n```\n2\n```')
        self.assertIsNotNone(error)
        _, error = suite_mod.extract_json('note\n```json\n{"a": 1}\n```')
        self.assertIsNotNone(error)
        _, error = suite_mod.extract_json('NaN')
        self.assertIsNotNone(error)
        _, error = suite_mod.extract_json(123)
        self.assertIsNotNone(error)

    def test_missing_and_duplicate_fail_explicitly(self):
        suite = suite_mod.build_suite()
        good = {t["id"]: json.dumps(t["expected"], ensure_ascii=False) for t in suite["tasks"]}
        dropped = dict(good)
        missing_id = suite["tasks"][0]["id"]
        del dropped[missing_id]
        report = suite_mod.score_suite(suite, dropped)
        self.assertEqual(report["totals"]["total"], 64)
        self.assertIn(missing_id, report["missing_ids"])
        case = next(c for c in report["cases"] if c["id"] == missing_id)
        self.assertFalse(case["passed"])
        self.assertEqual(case["error"], "missing_result")
        dup_id = suite["tasks"][1]["id"]
        listed = [{"task_id": tid, "response": text} for tid, text in good.items()]
        listed.append({"task_id": dup_id, "response": good[dup_id]})
        report = suite_mod.score_suite(suite, listed)
        self.assertIn(dup_id, report["duplicate_ids"])
        case = next(c for c in report["cases"] if c["id"] == dup_id)
        self.assertFalse(case["passed"])
        self.assertEqual(case["error"], "duplicate_result")

    def test_mapping_and_evaluator_formats(self):
        suite = suite_mod.build_suite()
        good = {t["id"]: json.dumps(t["expected"], ensure_ascii=False) for t in suite["tasks"]}
        report = suite_mod.score_suite(suite, good)
        self.assertEqual(report["totals"]["passed"], 64)
        evaluator = {"results": [{"name": tid, "output_text": text} for tid, text in good.items()]}
        report = suite_mod.score_suite(suite, evaluator)
        self.assertEqual(report["totals"]["passed"], 64)
        single = {"name": suite["tasks"][0]["id"],
                  "output_text": json.dumps(suite["tasks"][0]["expected"])}
        report = suite_mod.score_suite(suite, single)
        self.assertEqual(report["totals"]["total"], 64)
        self.assertEqual(len(report["missing_ids"]), 63)

    def test_object_key_order_and_unknown(self):
        suite = suite_mod.build_suite()
        target = next(t for t in suite["tasks"] if isinstance(t["expected"], dict))
        flipped = json.dumps({k: target["expected"][k] for k in reversed(list(target["expected"]))})
        good = {t["id"]: json.dumps(t["expected"], ensure_ascii=False) for t in suite["tasks"]}
        good[target["id"]] = flipped
        good["no-such-task"] = "1"
        report = suite_mod.score_suite(suite, good)
        case = next(c for c in report["cases"] if c["id"] == target["id"])
        self.assertTrue(case["passed"])
        self.assertIn("no-such-task", report["unknown_ids"])
        self.assertEqual(report["totals"]["total"], 64)

    def test_wilson_bounds(self):
        low, high = suite_mod.wilson_ci(32, 64)
        self.assertLessEqual(0.0, low)
        self.assertLessEqual(low, 0.5)
        self.assertLessEqual(0.5, high)
        self.assertLessEqual(high, 1.0)
        low, high = suite_mod.wilson_ci(64, 64)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)
        self.assertEqual(suite_mod.wilson_ci(0, 0), (0.0, 0.0))


class CliTests(unittest.TestCase):
    def test_create_and_score_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            suite_path = tmp / "suite.json"
            suite_mod.main(["create", "--output", str(suite_path)])
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertEqual(len(suite["tasks"]), 64)
            with self.assertRaises(SystemExit):
                suite_mod.main(["create", "--output", str(suite_path)])
            results_path = tmp / "results.json"
            good = {t["id"]: json.dumps(t["expected"], ensure_ascii=False) for t in suite["tasks"]}
            results_path.write_text(json.dumps(good), encoding="utf-8")
            report_path = tmp / "report.json"
            suite_mod.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                            "--output", str(report_path)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["passed"], 64)
            self.assertIn("by_category", report)
            self.assertIn("by_family", report)
            with self.assertRaises(SystemExit):
                suite_mod.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                                "--output", str(report_path)])


if __name__ == "__main__":
    unittest.main()
