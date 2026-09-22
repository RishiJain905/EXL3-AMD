"""Deterministic held-out MiMo validation fixtures: generator + scorer.

Stdlib only. Seed 20260922. Exactly 64 tasks (16 per category, 4 families
per category x 4 parameterizations). No calibration data, no model reads,
no code execution of model output, no external tool calls.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 20260922
SCHEMA = "mimo-validation-v1"
SYSTEM = "Follow the user instructions. Return only the requested JSON value."
QUALITY_MAX_TOKENS = 128

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_PROMPT_CHAR_LIMIT = 2000
_EXPECTED_CHAR_LIMIT = 600


def _reject_constant(value):
    raise ValueError("non-JSON constant: " + value)


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse_json(text):
    return json.loads(text, parse_constant=_reject_constant,
                      parse_float=_finite_float, object_pairs_hook=_unique_pairs)


# --------------------------------------------------------------------------
# code families: output prediction for small explicit programs
# --------------------------------------------------------------------------

def _gen_code_filter_reduce(rng, k):
    op = ("sumsq", "sum", "nonmatch", "sumcount")[k]
    d = rng.randint(2, 5)
    r = rng.randint(0, d - 1)
    values = [rng.randint(0, 20) for _ in range(8)]
    for _ in range(100):
        hits = sum(1 for x in values if x % d == r)
        if 1 <= hits <= 7:
            break
        values = [rng.randint(0, 20) for _ in range(8)]
    if op == "sumsq":
        expected = sum(x * x for x in values if x % d == r)
        body = (f"total = 0\nfor x in values:\n    if x % {d} == {r}:\n"
                "        total += x * x\nresult = total")
    elif op == "sum":
        expected = sum(x for x in values if x % d == r)
        body = (f"total = 0\nfor x in values:\n    if x % {d} == {r}:\n"
                "        total += x\nresult = total")
    elif op == "nonmatch":
        expected = sum(2 * x + 1 for x in values if x % d != r)
        body = (f"total = 0\nfor x in values:\n    if x % {d} != {r}:\n"
                "        total += 2 * x + 1\nresult = total")
    else:
        sel = [x for x in values if x % d == r]
        expected = sum(sel) * len(sel)
        body = (f"total = 0\ncount = 0\nfor x in values:\n    if x % {d} == {r}:\n"
                "        total += x\n        count += 1\nresult = total * count")
    prompt = ("Predict the exact output of this Python program.\n"
              f"values = {values}\nd = {d}\nr = {r}\n{body}\n"
              "What is the final value of `result`? "
              "Return only the requested JSON value (a JSON number).")
    return prompt, expected


_GROUP_POOLS = [
    ["red", "green", "blue", "amber"],
    ["alpha", "bravo", "cinder", "drift"],
    ["oak", "pine", "maple", "birch"],
    ["north", "south", "east", "west"],
]


def _gen_code_group_count(rng, k):
    pool = _GROUP_POOLS[k]
    n = 8 + k
    items = [rng.choice(pool) for _ in range(n)]
    for _ in range(50):
        if 2 <= len(set(items)) <= 4:
            break
        items = [rng.choice(pool) for _ in range(n)]
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    expected = [[key, items.count(key)] for key in seen]
    prompt = ("Predict the exact output of this Python program.\n"
              f"items = {items}\n"
              "counts = []\nseen = []\nfor item in items:\n"
              "    if item not in seen:\n"
              "        seen.append(item)\n"
              "        counts.append([item, items.count(item)])\n"
              "result = counts\n"
              "What is the final value of `result`? Return only the requested "
              "JSON value (a JSON array of [value, count] pairs in first-seen order).")
    return prompt, expected


def _gen_code_loop_trace(rng, k):
    a = rng.randint(0, 2)
    b = a + rng.randint(3, 5)
    c = rng.randint(0, 2)
    d = c + rng.randint(3, 5)
    m = rng.randint(2, 4)
    total = (b - a) * (d - c)
    pick_r = rng.randint(0, m - 1)

    def hits(r):
        if k == 2:
            return sum(1 for i in range(a, b) for j in range(c, d) if (i * j) % m == r)
        return sum(1 for i in range(a, b) for j in range(c, d) if (i + j) % m == r)

    r = pick_r
    if not 1 <= hits(r) <= total - 1:
        for cand in range(m):
            if 1 <= hits(cand) <= total - 1:
                r = cand
                break
    if k == 0:
        expected = sum(i + j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
        cond, acc = f"(i + j) % {m} == {r}", "total += i + j"
    elif k == 1:
        expected = sum(i * j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
        cond, acc = f"(i + j) % {m} == {r}", "total += i * j"
    elif k == 2:
        expected = sum(1 for i in range(a, b) for j in range(c, d) if (i * j) % m == r)
        cond, acc = f"(i * j) % {m} == {r}", "total += 1"
    else:
        expected = sum(i - j for i in range(a, b) for j in range(c, d) if (i + j) % m == r)
        cond, acc = f"(i + j) % {m} == {r}", "total += i - j"
    prompt = ("Predict the exact output of this Python program.\n"
              f"total = 0\nfor i in range({a}, {b}):\n"
              f"    for j in range({c}, {d}):\n"
              f"        if {cond}:\n            {acc}\nresult = total\n"
              "What is the final value of `result`? "
              "Return only the requested JSON value (a JSON number).")
    return prompt, expected


_DICT_POOLS = [
    ["a", "b", "c", "d"],
    ["x", "y", "z", "w"],
    ["p", "q", "r", "s"],
    ["m", "n", "k", "t"],
]


def _gen_code_dict_state(rng, k):
    keys = _DICT_POOLS[k]
    init_keys = keys[:3]
    state = {kk: rng.randint(1, 10) for kk in init_keys}
    cur = dict(state)
    newkey = keys[3]
    lines = []
    for step in range(5):
        if step == 2:
            s = rng.choice(sorted(cur))
            lines.append(f'state["{newkey}"] = state["{s}"] * 2')
            cur[newkey] = cur[s] * 2
            continue
        choice = rng.randint(0, 2)
        if choice == 0:
            t = rng.choice(sorted(cur))
            delta = rng.randint(1, 5)
            lines.append(f'state["{t}"] += {delta}')
            cur[t] += delta
        elif choice == 1:
            t = rng.choice(sorted(cur))
            s = rng.choice(sorted(cur))
            delta = rng.randint(1, 5)
            lines.append(f'state["{t}"] = state["{s}"] + {delta}')
            cur[t] = cur[s] + delta
        else:
            t = rng.choice(sorted(cur))
            s = rng.choice(sorted(cur))
            u = rng.choice(sorted(cur))
            lines.append(f'state["{t}"] = state["{s}"] + state["{u}"]')
            cur[t] = cur[s] + cur[u]
    init_repr = "{" + ", ".join(f'"{kk}": {state[kk]}' for kk in init_keys) + "}"
    prompt = ("Predict the exact output of this Python program.\n"
              f"state = {init_repr}\n" + "\n".join(lines) + "\nresult = state\n"
              "What is the final value of `result`? Return only the requested "
              "JSON value (a JSON object mapping keys to integers).")
    return prompt, dict(cur)


# --------------------------------------------------------------------------
# reasoning families: integer arithmetic, ratios, ordering, schedules
# --------------------------------------------------------------------------

def _gen_reasoning_inventory(rng, k):
    place, unit = (("depot", "crates"), ("library", "books"),
                   ("parts bin", "gears"), ("feed store", "sacks"))[k]
    kinds = ["received", "shipped", "damaged", "returned"]
    kinds = kinds[k:] + kinds[:k]
    start = rng.randint(60, 180)
    cur = start
    verbs = {"received": "Received", "shipped": "Shipped",
             "damaged": "Discarded as damaged", "returned": "Accepted as returns"}
    steps = []
    for kind in kinds:
        if kind in ("received", "returned"):
            amt = rng.randint(5, 40)
            cur += amt
        else:
            amt = rng.randint(5, min(40, max(5, cur - 10)))
            cur -= amt
        steps.append(f"{verbs[kind]} {amt} {unit}.")
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    prompt = (f"A {place} starts the week with {start} {unit}.\n{numbered}\n"
              f"How many {unit} remain at the end? "
              "Return only the requested JSON value (a JSON number).")
    return prompt, cur


def _gen_reasoning_ratio(rng, k):
    if k == 0:
        crates, boxes, parts = rng.randint(3, 6), rng.randint(4, 8), rng.randint(5, 9)
        removed = rng.randint(5, 30)
        prompt = (f"A shipment has {crates} crates. Each crate holds {boxes} boxes "
                  f"and each box holds {parts} parts. After unpacking, {removed} parts "
                  "are set aside. How many parts remain? "
                  "Return only the requested JSON value (a JSON number).")
        return prompt, crates * boxes * parts - removed
    if k == 1:
        a, b, units = rng.randint(2, 5), rng.randint(2, 5), rng.randint(3, 8)
        prompt = (f"An alloy mixes metal A and metal B in the ratio {a}:{b} by weight. "
                  f"A batch uses {(a + b) * units} kg in total. How many kg of each metal "
                  'are used? Return only the requested JSON value as an object '
                  '{"a": <kg of A>, "b": <kg of B>}.')
        return prompt, {"a": a * units, "b": b * units}
    if k == 2:
        serves, flour, mult = rng.randint(2, 4), rng.randint(100, 300), rng.randint(2, 4)
        prompt = (f"A recipe uses {flour} grams of flour to make {serves} servings. "
                  f"How many grams are needed to make {serves * mult} servings at the same "
                  "ratio? Return only the requested JSON value (a JSON number).")
        return prompt, flour * mult
    per_hour, hours, days = rng.randint(6, 12), rng.randint(3, 6), rng.randint(2, 4)
    defective = rng.randint(3, 15)
    prompt = (f"A machine makes {per_hour} items per hour. It runs {hours} hours per day "
              f"for {days} days. {defective} items fail inspection and are removed. "
              "How many good items remain? "
              "Return only the requested JSON value (a JSON number).")
    return prompt, per_hour * hours * days - defective


_ORDER_POOLS = [
    (["Amber", "Basil", "Cleo", "Dorian"], "Four runners finished a race with no ties"),
    (["Kite", "Lamp", "Maple", "North"], "Four jobs completed in a queue with no ties"),
    (["Ada", "Boris", "Cleo", "Dario", "Elif"], "Five stations were visited in order with no repeats"),
    (["Red", "Blue", "Green", "Amber", "Violet"], "Five parcels were delivered in order with no ties"),
]


def _gen_reasoning_order(rng, k):
    names, desc = _ORDER_POOLS[k]
    perm = list(names)
    rng.shuffle(perm)
    constraints = [f"{perm[i]} came before {perm[i + 1]}." for i in range(len(perm) - 1)]
    constraints.append(f"{perm[0]} came before {perm[2]}.")
    constraints.append(f"{perm[0]} was not last.")
    constraints.append(f"{perm[-1]} was not first.")
    rng.shuffle(constraints)
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(constraints))
    prompt = (f"{desc}. The following statements are all true:\n{numbered}\n"
              "List the full order from first to last as a JSON array of quoted names. "
              "Return only the requested JSON value.")
    return prompt, list(perm)


_SCHEDULE_DAGS = [
    [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
    [("A", "C"), ("B", "C"), ("C", "D")],
    [("A", "B"), ("A", "C"), ("B", "D")],
    [("A", "D"), ("B", "D"), ("C", "D")],
]


def _gen_reasoning_schedule(rng, k):
    edges = _SCHEDULE_DAGS[k]
    tasks = ["A", "B", "C", "D"]
    durations = {t: rng.randint(2, 6) for t in tasks}
    finish = {}
    for t in tasks:
        preds = [s for s, e in edges if e == t]
        start = max((finish[p] for p in preds), default=0)
        finish[t] = start + durations[t]
    makespan = max(finish.values())
    lines = "\n".join(
        f"- Task {t}: takes {durations[t]} hours; prerequisites: "
        f"{', '.join(sorted(s for s, e in edges if e == t)) or 'none'}."
        for t in tasks)
    prompt = ("Four tasks can run in parallel on unlimited machines. A task may start "
              f"only after all its prerequisites finish. Machines are available from time 0; tasks without prerequisites may start then.\n{lines}\n"
              "What is the earliest time (in hours) when all tasks are complete? "
              "Return only the requested JSON value (a JSON number).")
    return prompt, makespan


# --------------------------------------------------------------------------
# tools families: fictional choose-a-tool + JSON normalization/extraction
# --------------------------------------------------------------------------

_SINGLE_CATALOG = [
    {"name": "shelf_lookup", "description": "Look up a shelf location.",
     "parameters": {"shelf_id": "string shelf code"}},
    {"name": "restock_order", "description": "Order restock of an SKU.",
     "parameters": {"sku": "string", "quantity": "integer"}},
    {"name": "lamp_set", "description": "Set lamp brightness.",
     "parameters": {"room": "string", "brightness": "integer 0-100"}},
    {"name": "log_note", "description": "Record a tagged note.",
     "parameters": {"tag": "string", "message": "string"}},
    {"name": "valve_close", "description": "Close a valve.",
     "parameters": {"valve_id": "string"}},
    {"name": "timer_start", "description": "Start a timer.",
     "parameters": {"label": "string", "seconds": "integer"}},
]


def _single_args(rng, target):
    if target == "restock_order":
        return {"sku": f"SKU-{rng.randint(1000, 9999)}", "quantity": rng.randint(5, 50)}
    if target == "lamp_set":
        return {"room": rng.choice(["kitchen", "study", "garage", "loft"]),
                "brightness": rng.randint(10, 90)}
    if target == "log_note":
        return {"tag": rng.choice(["audit", "shift", "safety"]),
                "message": f"check {rng.choice(['pump', 'filter', 'gauge'])} {rng.randint(1, 9)}"}
    return {"label": rng.choice(["bake", "soak", "cooldown"]), "seconds": rng.randint(60, 3600)}


def _single_request(target, args):
    if target == "restock_order":
        return f"Order {args['quantity']} units of {args['sku']}."
    if target == "lamp_set":
        return f"Set the {args['room']} lamp brightness to {args['brightness']}."
    if target == "log_note":
        return f"Record a {args['tag']} note saying {args['message']!r}."
    return f"Start a {args['label']} timer for {args['seconds']} seconds."


def _gen_tools_choose_single(rng, k):
    target = ("restock_order", "lamp_set", "log_note", "timer_start")[k]
    args = _single_args(rng, target)
    by_name = {t["name"]: t for t in _SINGLE_CATALOG}
    others = [t for t in _SINGLE_CATALOG if t["name"] != target]
    rng.shuffle(others)
    offered = [by_name[target]] + others[:2]
    rng.shuffle(offered)
    prompt = ("Available tools (fictional; do not execute anything):\n"
              f"{json.dumps(offered, indent=2)}\n"
              f"Request: {_single_request(target, args)}\n"
              "Choose the single tool and arguments that satisfy the request. "
              "Return only the requested JSON value as "
              '{"name": <tool name>, "arguments": {<args>}}.')
    return prompt, {"name": target, "arguments": args}


def _multi_spec(rng, k):
    if k == 0:
        stops = rng.sample(["Ashford", "Bexley", "Corby", "Dover", "Ely"], 2)
        avoid = rng.choice([True, False])
        hours = rng.randint(2, 8)
        tool = {"name": "route_plan", "description": "Plan a fictional route.",
                "parameters": {"stops": "array of string",
                               "options": "{avoid_tolls: boolean, max_hours: integer}"}}
        args = {"stops": stops, "options": {"avoid_tolls": avoid, "max_hours": hours}}
        tolls = "avoiding tolls" if avoid else "allowing tolls"
        req = (f"Plan a route through {stops[0]} then {stops[1]}, {tolls}, "
               f"with at most {hours} hours.")
        return tool, args, req
    if k == 1:
        ids = [f"P-{rng.randint(100, 999)}", f"P-{rng.randint(100, 999)}"]
        q0, q1 = rng.randint(1, 20), rng.randint(1, 20)
        tool = {"name": "batch_update", "description": "Apply fictional stock changes.",
                "parameters": {"changes": "array of {id: string, quantity: integer}"}}
        args = {"changes": [{"id": ids[0], "quantity": q0}, {"id": ids[1], "quantity": q1}]}
        req = f"Set {ids[0]} to {q0} and {ids[1]} to {q1} in one batch."
        return tool, args, req
    if k == 2:
        title = f"Shift survey {rng.randint(1, 9)}"
        qs = [f"Rate {rng.choice(['pump', 'filter', 'gauge'])} {i + 1}" for i in range(2)]
        anon = rng.choice([True, False])
        tool = {"name": "survey_create", "description": "Create a fictional survey.",
                "parameters": {"title": "string", "questions": "array of string",
                               "anonymous": "boolean"}}
        args = {"title": title, "questions": qs, "anonymous": anon}
        mode = "anonymous" if anon else "named"
        req = (f"Create {mode} survey {title!r} with questions {qs[0]!r} and {qs[1]!r}.")
        return tool, args, req
    field = rng.choice(["stock", "score", "level"])
    vals = sorted(rng.sample(range(1, 30), 3))
    mode = rng.choice(["any", "all"])
    tool = {"name": "filter_set", "description": "Set a fictional filter.",
            "parameters": {"field": "string", "values": "array of integer",
                           "mode": "string any|all"}}
    args = {"field": field, "values": vals, "mode": mode}
    req = f"Filter {field} to {mode} of {vals[0]}, {vals[1]} and {vals[2]}."
    return tool, args, req


def _gen_tools_choose_multi(rng, k):
    tool, args, req = _multi_spec(rng, k)
    others = [t for t in _SINGLE_CATALOG if t["name"] not in (tool["name"],)]
    rng.shuffle(others)
    offered = [tool] + others[:2]
    rng.shuffle(offered)
    prompt = ("Available tools (fictional; do not execute anything):\n"
              f"{json.dumps(offered, indent=2)}\n"
              f"Request: {req}\n"
              "Choose the single tool and arguments that satisfy the request. "
              "Return only the requested JSON value as "
              '{"name": <tool name>, "arguments": {<args>}}.')
    return prompt, {"name": tool["name"], "arguments": args}


def _gen_tools_normalize(rng, k):
    if k == 0:
        ident, name = rng.randint(10, 99), rng.choice(["Widget", "Gasket", "Flange"])
        tags = rng.sample(["red", "blue", "steel", "small", "large"], 3)
        raw = {"ID": str(ident), "Name": f"  {name} ", "tags": tags, "extra": "drop me"}
        expected = {"id": ident, "name": name, "tags": sorted(tags)}
        rules = ("1. Rename ID to id and convert the string to an integer.\n"
                 "2. Rename Name to name and trim surrounding spaces.\n"
                 "3. Sort tags alphabetically.\n4. Drop any other keys.")
    elif k == 1:
        code, reading = f"S-{rng.randint(100, 999)}", rng.randint(10, 99)
        raw = {"Sensor": code, " Reading ": str(reading), "unit": " C ", "debug": "x"}
        expected = {"sensor": code, "reading": reading, "unit": "C"}
        rules = ("1. Rename Sensor to sensor.\n"
                 "2. Rename ' Reading ' to reading and convert the string to an integer.\n"
                 "3. Trim spaces from unit.\n4. Drop any other keys.")
    elif k == 2:
        uid = rng.randint(1000, 9999)
        user = rng.choice(["ana", "ben", "cid", "dee"])
        roles = rng.sample(["reader", "writer", "auditor", "viewer"], 2)
        raw = {"user_id": str(uid), "email": f"  {user}@example.test ", "roles": roles,
               "session": "drop"}
        expected = {"user_id": uid, "email": f"{user}@example.test", "roles": sorted(roles)}
        rules = ("1. Convert user_id from string to integer.\n"
                 "2. Trim spaces from email.\n"
                 "3. Sort roles alphabetically.\n4. Drop any other keys.")
    else:
        order, qty = rng.randint(500, 999), rng.randint(1, 25)
        prio = rng.choice(["High", "Low"])
        raw = {"order": str(order), "qty": str(qty), "priority": f"  {prio} ",
               "note": "drop"}
        expected = {"order": order, "qty": qty, "priority": prio}
        rules = ("1. Convert order and qty from strings to integers.\n"
                 "2. Trim spaces from priority.\n3. Drop any other keys.")
    prompt = (f"Normalize this JSON value:\n{json.dumps(raw)}\nRules:\n{rules}\n"
              "Return only the requested JSON value (the normalized JSON object).")
    return prompt, expected


def _gen_tools_extract(rng, k):
    if k == 0:
        oid = rng.randint(1000, 9999)
        name = rng.choice(["Ana", "Ben", "Cid", "Dee"])
        item1, item2 = rng.sample(["bolt", "nut", "washer", "pin"], 2)
        q1, q2 = rng.randint(1, 9), rng.randint(1, 9)
        prio = rng.choice(["high", "low"])
        record = (f"Order #{oid} | customer: {name} | items: {q1}x {item1}, "
                  f"{q2}x {item2} | priority: {prio}")
        expected = {"order": oid, "customer": name,
                    "items": [{"name": item1, "qty": q1}, {"name": item2, "qty": q2}],
                    "priority": prio}
        schema = ('Extract {"order": integer, "customer": string, '
                  '"items": [{"name": string, "qty": integer}] in listed order, '
                  '"priority": string}.')
    elif k == 1:
        day, level = rng.randint(1, 9), rng.choice(["INFO", "WARN"])
        service, code = rng.choice(["pump", "valve"]), f"E{rng.randint(100, 999)}"
        retries, lat = rng.randint(0, 5), rng.randint(5, 400)
        record = (f"[2026-03-0{day} 10:15:00] {level} {service}: {code} "
                  f"retries={retries} latency_ms={lat}")
        expected = {"date": f"2026-03-0{day}", "level": level, "service": service,
                    "code": code, "retries": retries, "latency_ms": lat}
        schema = ('Extract {"date": string YYYY-MM-DD, "level": string, "service": string, '
                  '"code": string, "retries": integer, "latency_ms": integer}.')
    elif k == 2:
        sku = f"SKU-{rng.randint(100, 999)}"
        count = rng.randint(5, 60)
        bins = [f"B{rng.randint(1, 9)}", f"B{rng.randint(1, 9)}"]
        flag = rng.choice(["yes", "no"])
        record = f"{sku} count={count} bins={bins[0]},{bins[1]} flag={flag}"
        expected = {"sku": sku, "count": count, "bins": bins, "flag": flag}
        schema = ('Extract {"sku": string, "count": integer, "bins": [string, string] '
                  'in listed order, "flag": string}.')
    else:
        tid, sev = rng.randint(100, 999), rng.choice(["S1", "S3"])
        system = rng.choice(["pump", "press"])
        person = rng.choice(["Ana", "Ben", "Cid"])
        hours = rng.randint(1, 12)
        record = f"Ticket {tid} [{sev}] {system} assignee={person} hours={hours}"
        expected = {"ticket": tid, "severity": sev, "system": system,
                    "assignee": person, "hours": hours}
        schema = ('Extract {"ticket": integer, "severity": string, "system": string, '
                  '"assignee": string, "hours": integer}.')
    prompt = (f"Record:\n{record}\n{schema}\n"
              "Return only the requested JSON value (the extracted JSON object).")
    return prompt, expected


# --------------------------------------------------------------------------
# general families: grounded extraction/sorting/filtering/aggregation
# --------------------------------------------------------------------------

_STATIONS = ["S-1", "S-2", "S-3", "S-4", "S-5", "S-6"]
_CITIES = ["Ashford", "Bexley", "Corby", "Dover", "Ely", "Fenton"]


def _gen_general_lookup(rng, k):
    if k == 0:
        codes = rng.sample(_STATIONS, 5)
        cities = rng.sample(_CITIES, 5)
        pick = rng.randrange(5)
        lines = "\n".join(f"{c} | city: {city}" for c, city in zip(codes, cities))
        prompt = (f"Station roster (invented facts):\n{lines}\n"
                  f"What city hosts station {codes[pick]}? "
                  "Return only the requested JSON value (a JSON string).")
        return prompt, cities[pick]
    if k == 1:
        names = rng.sample(["Ana", "Ben", "Cid", "Dee", "Eli", "Fay"], 5)
        roles = rng.sample(["pilot", "medic", "clerk", "cook", "guard", "tech"], 5)
        pick = rng.randrange(5)
        lines = "\n".join(f"{n} | role: {r}" for n, r in zip(names, roles))
        prompt = (f"Crew roster (invented facts):\n{lines}\n"
                  f"What is the role of {names[pick]}? "
                  "Return only the requested JSON value (a JSON string).")
        return prompt, roles[pick]
    if k == 2:
        skus = [f"SKU-{n}" for n in rng.sample(range(100, 999), 5)]
        stocks = rng.sample(range(5, 80), 5)
        pick = rng.randrange(5)
        lines = "\n".join(f"{s} | stock: {v}" for s, v in zip(skus, stocks))
        prompt = (f"Stock table (invented facts):\n{lines}\n"
                  f"What is the stock of {skus[pick]}? "
                  "Return only the requested JSON value (a JSON number).")
        return prompt, stocks[pick]
    titles = [f"Log {n}" for n in rng.sample(range(10, 99), 5)]
    shelves = [f"H-{rng.randint(1, 4)}" for _ in range(5)]
    pick = rng.randrange(5)
    lines = "\n".join(f"{t} | shelf: {s}" for t, s in zip(titles, shelves))
    prompt = (f"Catalog (invented facts):\n{lines}\n"
              f"Which shelf holds {titles[pick]}? "
              "Return only the requested JSON value (a JSON string).")
    return prompt, shelves[pick]


def _gen_general_sort(rng, k):
    if k == 0:
        codes = rng.sample(_STATIONS, 6)
        status = ["active" if i % 2 == 0 else "spare" for i in range(6)]
        pairs = list(zip(codes, status))
        rng.shuffle(pairs)
        lines = "\n".join(f"{c} | status: {s}" for c, s in pairs)
        expected = sorted(c for c, s in pairs if s == "active")
        prompt = (f"Station roster (invented facts):\n{lines}\n"
                  "List the codes with status active, sorted alphabetically. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    if k == 1:
        skus = [f"SKU-{n}" for n in rng.sample(range(100, 999), 5)]
        stocks = rng.sample(range(5, 80), 5)
        lines = "\n".join(f"{s} | stock: {v}" for s, v in zip(skus, stocks))
        expected = [s for s, _ in sorted(zip(skus, stocks), key=lambda p: p[1])]
        prompt = (f"Stock table (invented facts):\n{lines}\n"
                  "List the SKUs from lowest to highest stock. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    if k == 2:
        names = rng.sample(["Ana", "Ben", "Cid", "Dee", "Eli", "Fay"], 5)
        ages = rng.sample(range(20, 60), 5)
        lines = "\n".join(f"{n} | age: {a}" for n, a in zip(names, ages))
        expected = [n for n, _ in sorted(zip(names, ages), key=lambda p: -p[1])]
        prompt = (f"Crew ages (invented facts):\n{lines}\n"
                  "List the names from oldest to youngest. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    titles = [f"Log {n}" for n in rng.sample(range(10, 99), 6)]
    shelves = ["H-1" if i % 2 == 0 else "H-2" for i in range(6)]
    pairs = list(zip(titles, shelves))
    rng.shuffle(pairs)
    lines = "\n".join(f"{t} | shelf: {s}" for t, s in pairs)
    expected = sorted(t for t, s in pairs if s == "H-1")
    prompt = (f"Catalog (invented facts):\n{lines}\n"
              "List the titles on shelf H-1, sorted alphabetically. "
              "Return only the requested JSON value (a JSON array of strings).")
    return prompt, expected


def _gen_general_filter(rng, k):
    if k == 0:
        ids = [f"G-{i}" for i in rng.sample(range(10, 99), 6)]
        readings = rng.sample(range(10, 90), 6)
        thresh = sorted(readings)[2]
        lines = "\n".join(f"{i} | reading: {v}" for i, v in zip(ids, readings))
        expected = [i for i, v in zip(ids, readings) if v >= thresh]
        prompt = (f"Gauge readings (invented facts):\n{lines}\n"
                  f"List the gauge IDs with reading at least {thresh}, in roster order. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    if k == 1:
        names = rng.sample(["Ana", "Ben", "Cid", "Dee", "Eli", "Fay"], 6)
        shifts = ["night" if i % 2 == 0 else "day" for i in range(6)]
        pairs = list(zip(names, shifts))
        rng.shuffle(pairs)
        lines = "\n".join(f"{n} | shift: {s}" for n, s in pairs)
        expected = [n for n, s in pairs if s == "night"]
        prompt = (f"Crew shifts (invented facts):\n{lines}\n"
                  "List the night-shift crew in roster order. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    if k == 2:
        skus = [f"SKU-{n}" for n in rng.sample(range(100, 999), 6)]
        stocks = rng.sample(range(5, 80), 6)
        thresh = sorted(stocks)[3]
        lines = "\n".join(f"{s} | stock: {v}" for s, v in zip(skus, stocks))
        expected = [s for s, v in zip(skus, stocks) if v < thresh]
        prompt = (f"Stock table (invented facts):\n{lines}\n"
                  f"List the SKUs with stock below {thresh}, in table order. "
                  "Return only the requested JSON value (a JSON array of strings).")
        return prompt, expected
    titles = [f"Log {n}" for n in rng.sample(range(10, 99), 6)]
    pages = rng.sample(range(20, 200), 6)
    thresh = sorted(pages)[2]
    lines = "\n".join(f"{t} | pages: {p}" for t, p in zip(titles, pages))
    expected = [t for t, p in zip(titles, pages) if p > thresh]
    prompt = (f"Catalog (invented facts):\n{lines}\n"
              f"List the titles with more than {thresh} pages, in catalog order. "
              "Return only the requested JSON value (a JSON array of strings).")
    return prompt, expected


def _gen_general_aggregate(rng, k):
    if k == 0:
        skus = [f"SKU-{n}" for n in rng.sample(range(100, 999), 6)]
        stocks = rng.sample(range(5, 60), 6)
        flags = ["active" if i % 2 == 0 else "spare" for i in range(6)]
        rows = list(zip(skus, stocks, flags))
        rng.shuffle(rows)
        lines = "\n".join(f"{s} | stock: {v} | status: {f}" for s, v, f in rows)
        sel = [v for _, v, f in rows if f == "active"]
        prompt = (f"Stock table (invented facts):\n{lines}\n"
                  'For status active, return {"count": <rows>, "total": <stock sum>}. '
                  "Return only the requested JSON value.")
        return prompt, {"count": len(sel), "total": sum(sel)}
    if k == 1:
        ids = [f"G-{i}" for i in rng.sample(range(10, 99), 5)]
        readings = rng.sample(range(10, 90), 5)
        lines = "\n".join(f"{i} | reading: {v}" for i, v in zip(ids, readings))
        prompt = (f"Gauge readings (invented facts):\n{lines}\n"
                  'Return {"min": <lowest reading>, "max": <highest reading>}. '
                  "Return only the requested JSON value.")
        return prompt, {"min": min(readings), "max": max(readings)}
    if k == 2:
        names = rng.sample(["Ana", "Ben", "Cid", "Dee", "Eli", "Fay"], 6)
        hours = rng.sample(range(1, 12), 6)
        shifts = ["night" if i % 2 == 0 else "day" for i in range(6)]
        rows = list(zip(names, hours, shifts))
        rng.shuffle(rows)
        lines = "\n".join(f"{n} | hours: {h} | shift: {s}" for n, h, s in rows)
        sel = [h for _, h, s in rows if s == "night"]
        prompt = (f"Crew hours (invented facts):\n{lines}\n"
                  'For night shift, return {"count": <rows>, "total_hours": <sum>}. '
                  "Return only the requested JSON value.")
        return prompt, {"count": len(sel), "total_hours": sum(sel)}
    titles = [f"Log {n}" for n in rng.sample(range(10, 99), 6)]
    pages = rng.sample(range(20, 200), 6)
    shelves = ["H-1" if i % 2 == 0 else "H-2" for i in range(6)]
    rows = list(zip(titles, pages, shelves))
    rng.shuffle(rows)
    lines = "\n".join(f"{t} | pages: {p} | shelf: {s}" for t, p, s in rows)
    sel = [p for _, p, s in rows if s == "H-1"]
    prompt = (f"Catalog (invented facts):\n{lines}\n"
              'For shelf H-1, return {"count": <rows>, "pages": <page sum>}. '
              "Return only the requested JSON value.")
    return prompt, {"count": len(sel), "pages": sum(sel)}


_FAMILIES = (
    ("code", "code_filter_reduce", _gen_code_filter_reduce),
    ("code", "code_group_count", _gen_code_group_count),
    ("code", "code_loop_trace", _gen_code_loop_trace),
    ("code", "code_dict_state", _gen_code_dict_state),
    ("reasoning", "reasoning_inventory", _gen_reasoning_inventory),
    ("reasoning", "reasoning_ratio", _gen_reasoning_ratio),
    ("reasoning", "reasoning_order", _gen_reasoning_order),
    ("reasoning", "reasoning_schedule", _gen_reasoning_schedule),
    ("tools", "tools_choose_single", _gen_tools_choose_single),
    ("tools", "tools_choose_multi", _gen_tools_choose_multi),
    ("tools", "tools_normalize", _gen_tools_normalize),
    ("tools", "tools_extract", _gen_tools_extract),
    ("general", "general_lookup", _gen_general_lookup),
    ("general", "general_sort", _gen_general_sort),
    ("general", "general_filter", _gen_general_filter),
    ("general", "general_aggregate", _gen_general_aggregate),
)


def build_suite(seed=SEED):
    rng = random.Random(seed)
    tasks = []
    for category, family, gen in _FAMILIES:
        for k in range(4):
            prompt, expected = gen(rng, k)
            tasks.append({"id": f"{family}-{k:02d}", "category": category,
                          "family": family, "prompt": prompt, "expected": expected})
    return {
        "schema": SCHEMA,
        "seed": seed,
        "system": SYSTEM,
        "quality_max_tokens": QUALITY_MAX_TOKENS,
        "claim_limits": ("Small procedural validation fixtures, not a public benchmark, "
                         "not an equivalence test, not BF16 fidelity. No code execution "
                         "of model output and no real tool-use success claim."),
        "tasks": tasks,
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def extract_json(text):
    if not isinstance(text, str):
        return None, "response_not_string"
    fences = _FENCE_RE.findall(text)
    if len(fences) > 1:
        return None, "multiple_fenced_blocks"
    if len(fences) == 1:
        match = _FENCE_RE.search(text)
        if match.start() is None:
            return None, "fence_error"
        before, after = text[:match.start()], text[match.end():]
        if before.strip() or after.strip():
            return None, "prose_around_fence"
        inner = match.group(0)[3:-3].strip()
        if inner[:4].lower() == "json" and (len(inner) == 4 or inner[4] in " \t\r\n"):
            inner = inner[4:].strip()
        try:
            return _parse_json(inner), None
        except ValueError as exc:
            return None, f"invalid_json_in_fence: {exc}"
    try:
        return _parse_json(text.strip()), None
    except ValueError as exc:
        return None, f"invalid_json: {exc}"


def json_equal(expected, actual):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(json_equal(expected[k], actual[k]) for k in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            json_equal(a, b) for a, b in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        if isinstance(expected, int) and isinstance(actual, float):
            return actual.is_integer() and expected == int(actual)
        if isinstance(expected, float) and isinstance(actual, int):
            return expected.is_integer() and int(expected) == actual
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def wilson_ci(passes, total, z=1.96):
    if total <= 0:
        return (0.0, 0.0)
    p = passes / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    delta = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (center - delta) / denom), min(1.0, (center + delta) / denom))


def _result_pairs(results_obj):
    """Return ([(task_id, response_text)], format_error)."""
    if isinstance(results_obj, dict) and isinstance(results_obj.get("results"), list):
        pairs = []
        for entry in results_obj["results"]:
            if not isinstance(entry, dict):
                return [], "result_entry_not_object"
            task_id = entry.get("name", entry.get("id", entry.get("task_id")))
            text = entry.get("output_text", entry.get("response_text",
                             entry.get("response", entry.get("text"))))
            if not isinstance(task_id, str) or not isinstance(text, str):
                return [], "result_entry_missing_name_or_text"
            pairs.append((task_id, text))
        return pairs, None
    if isinstance(results_obj, dict) and isinstance(results_obj.get("name"), str):
        text = results_obj.get("output_text", results_obj.get("response_text",
                               results_obj.get("response", results_obj.get("text"))))
        if not isinstance(text, str):
            return [], "single_result_missing_text"
        return [(results_obj["name"], text)], None
    if isinstance(results_obj, list):
        pairs = []
        for entry in results_obj:
            if not isinstance(entry, dict):
                return [], "result_entry_not_object"
            task_id = entry.get("task_id", entry.get("id", entry.get("name")))
            text = entry.get("response", entry.get("response_text",
                             entry.get("output_text", entry.get("text"))))
            if not isinstance(task_id, str) or not isinstance(text, str):
                return [], "result_entry_missing_name_or_text"
            pairs.append((task_id, text))
        return pairs, None
    if isinstance(results_obj, dict):
        pairs = []
        for key, value in results_obj.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return [], "mapping_results_must_be_str_to_str"
            pairs.append((key, value))
        return pairs, None
    return [], "unrecognized_results_format"


def score_suite(suite, results_obj):
    tasks = suite.get("tasks", [])
    pairs, format_error = _result_pairs(results_obj)
    counts = Counter(task_id for task_id, _ in pairs)
    duplicates = sorted(t for t, c in counts.items() if c > 1)
    first = {}
    for task_id, text in pairs:
        first.setdefault(task_id, text)
    by_id = {t["id"]: t for t in tasks}
    unknown = sorted(t for t in first if t not in by_id)
    cases = []
    for task in tasks:
        task_id = task["id"]
        if format_error is not None:
            cases.append({"id": task_id, "category": task.get("category"),
                          "family": task.get("family"), "passed": False,
                          "parse_error": None, "error": format_error,
                          "expected": task.get("expected"), "actual": None})
            continue
        if task_id in duplicates:
            cases.append({"id": task_id, "category": task.get("category"),
                          "family": task.get("family"), "passed": False,
                          "parse_error": None, "error": "duplicate_result",
                          "expected": task.get("expected"), "actual": None})
            continue
        if task_id not in first:
            cases.append({"id": task_id, "category": task.get("category"),
                          "family": task.get("family"), "passed": False,
                          "parse_error": None, "error": "missing_result",
                          "expected": task.get("expected"), "actual": None})
            continue
        actual, parse_error = extract_json(first[task_id])
        if parse_error is not None:
            cases.append({"id": task_id, "category": task.get("category"),
                          "family": task.get("family"), "passed": False,
                          "parse_error": parse_error, "error": "parse_error",
                          "expected": task.get("expected"), "actual": None})
            continue
        passed = json_equal(task.get("expected"), actual)
        cases.append({"id": task_id, "category": task.get("category"),
                      "family": task.get("family"), "passed": passed,
                      "parse_error": None,
                      "error": None if passed else "mismatch",
                      "expected": task.get("expected"), "actual": actual})

    def summarize(items):
        passed = sum(1 for c in items if c["passed"])
        total = len(items)
        low, high = wilson_ci(passed, total)
        return {"passed": passed, "total": total,
                "rate": (passed / total) if total else 0.0,
                "wilson_low": low, "wilson_high": high}

    by_category = {}
    by_family = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
        by_family.setdefault(case["family"], []).append(case)
    digest = hashlib.sha256(
        json.dumps(suite, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "schema": SCHEMA,
        "suite_sha256": digest,
        "totals": summarize(cases),
        "by_category": {k: summarize(v) for k, v in sorted(by_category.items())},
        "by_family": {k: summarize(v) for k, v in sorted(by_family.items())},
        "missing_ids": sorted(t["id"] for t in tasks if t["id"] not in first),
        "duplicate_ids": duplicates,
        "unknown_ids": unknown,
        "format_error": format_error,
        "cases": cases,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _refuse_overwrite(path):
    if Path(path).exists():
        print(f"refusing to overwrite existing file: {path}", file=sys.stderr)
        raise SystemExit(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Generate the deterministic 64-task suite.")
    create.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Score results against a suite.")
    score.add_argument("--suite", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        _refuse_overwrite(args.output)
        suite = build_suite()
        args.output.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"wrote {len(suite['tasks'])} tasks to {args.output}")
    elif args.command == "score":
        _refuse_overwrite(args.output)
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        results_obj = json.loads(args.results.read_text(encoding="utf-8"))
        report = score_suite(suite, results_obj)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        totals = report["totals"]
        print(f"passed {totals['passed']}/{totals['total']} "
              f"rate={totals['rate']:.3f} "
              f"wilson=[{totals['wilson_low']:.3f},{totals['wilson_high']:.3f}]")


if __name__ == "__main__":
    main()
