"""Post-hoc exploratory thinking-enabled supplement: builder + scorer.

Stdlib only at import. The `build` subcommand imports transformers lazily
(Main's registered environment); `score` and all helpers stay CPU-light.
Model output is parsed as data only; never executed.

Scoring accepts either strict direct JSON with NO think markers, OR exactly
ONE well-formed LEADING <think>...</think> block followed by strict JSON.
Final-answer semantics (duplicate-key / nonfinite rejection, single clean
fenced block) delegate to scripts/mimo_validation_suite.py unchanged.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

THINKING_MAX_TOKENS = 2048
CONTEXT = 4096
CACHE = "f16"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
OPEN = "<think>"
CLOSE = "</think>"
_ASCII_WS = " \t\n\r\f\v"

SUITE_A = "validation-suite-reasoning-s1a.json"
SUITE_B = "validation-suite-reasoning-s1b.json"
PROTOCOL_A = "evaluation-protocol-reasoning-s1a.json"
PROTOCOL_B = "evaluation-protocol-reasoning-s1b.json"
MANIFEST = "manifest.json"


def _load_base():
    path = Path(__file__).resolve().with_name("mimo_validation_suite.py")
    spec = importlib.util.spec_from_file_location("mimo_validation_suite_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()


# --------------------------------------------------------------------------
# thinking-prefix parser
# --------------------------------------------------------------------------

def strip_thinking_block(text):
    """Return (answer_text|None, reasoning_chars|None, error|None).

    No markers anywhere -> direct-JSON path: returns (text, None, None).
    Otherwise exactly one leading <think>...</think> block is required;
    the answer after the block is returned unmodified for strict parsing.
    """
    if not isinstance(text, str):
        return None, None, "response_not_string"
    has_open = OPEN in text
    has_close = CLOSE in text
    if not has_open and not has_close:
        return text, None, None
    if has_close and not has_open:
        return None, None, "thinking_block_unbalanced"
    start = text.find(OPEN)
    if text[:start].strip(_ASCII_WS):
        return None, None, "thinking_block_misplaced"
    end = text.find(CLOSE, start + len(OPEN))
    if end < 0:
        return None, None, "thinking_block_unclosed"
    reasoning = text[start + len(OPEN):end]
    answer = text[end + len(CLOSE):]
    if OPEN in reasoning or OPEN in answer or CLOSE in answer:
        return None, None, "thinking_block_extra_markers"
    return answer, len(reasoning), None


def score_suite_thinking(suite, results_obj):
    """Mirror base score_suite with a think-block pre-strip per result."""
    tasks = suite.get("tasks", [])
    pairs, format_error = _base._result_pairs(results_obj)
    counts = {}
    for task_id, _ in pairs:
        counts[task_id] = counts.get(task_id, 0) + 1
    duplicates = sorted(t for t, c in counts.items() if c > 1)
    first = {}
    for task_id, text in pairs:
        first.setdefault(task_id, text)
    by_id = {t["id"]: t for t in tasks}
    unknown = sorted(t for t in first if t not in by_id)
    cases = []
    for task in tasks:
        task_id = task["id"]
        skeleton = {"id": task_id, "category": task.get("category"),
                    "family": task.get("family"), "expected": task.get("expected")}
        if format_error is not None:
            cases.append({**skeleton, "passed": False, "parse_error": None,
                          "strip_error": None, "reasoning_chars": None,
                          "error": format_error, "actual": None})
            continue
        if task_id in duplicates:
            cases.append({**skeleton, "passed": False, "parse_error": None,
                          "strip_error": None, "reasoning_chars": None,
                          "error": "duplicate_result", "actual": None})
            continue
        if task_id not in first:
            cases.append({**skeleton, "passed": False, "parse_error": None,
                          "strip_error": None, "reasoning_chars": None,
                          "error": "missing_result", "actual": None})
            continue
        answer, reasoning_chars, strip_error = strip_thinking_block(first[task_id])
        if strip_error is not None:
            cases.append({**skeleton, "passed": False, "parse_error": None,
                          "strip_error": strip_error, "reasoning_chars": None,
                          "error": "thinking_parse_error", "actual": None})
            continue
        actual, parse_error = _base.extract_json(answer)
        if parse_error is not None:
            cases.append({**skeleton, "passed": False, "parse_error": parse_error,
                          "strip_error": None, "reasoning_chars": reasoning_chars,
                          "error": "parse_error", "actual": None})
            continue
        passed = _base.json_equal(task.get("expected"), actual)
        cases.append({**skeleton, "passed": passed, "parse_error": None,
                      "strip_error": None, "reasoning_chars": reasoning_chars,
                      "error": None if passed else "mismatch", "actual": actual})

    def summarize(items):
        passed = sum(1 for c in items if c["passed"])
        total = len(items)
        low, high = _base.wilson_ci(passed, total)
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
        "schema": _base.SCHEMA,
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
# supplement builder (pure helpers; tokenizer injected)
# --------------------------------------------------------------------------

def select_thinking_subset(tasks):
    """First task of each family in suite order; asserts 16 x -00 ids."""
    selected = []
    seen = set()
    for task in tasks:
        family = task.get("family")
        if family not in seen:
            seen.add(family)
            selected.append(copy.deepcopy(task))
    if len(selected) != 16:
        raise ValueError(f"expected 16 families, got {len(selected)}")
    for task in selected:
        if not task.get("id", "").endswith("-00"):
            raise ValueError("supplement subset must be first task per family (-00): "
                             + repr(task.get("id")))
    return selected


def split_batches(selected):
    """Fixed 8+8 split: batch A first 8, batch B last 8."""
    if len(selected) != 16:
        raise ValueError(f"expected 16 selected tasks, got {len(selected)}")
    return selected[:8], selected[8:]


def render_thinking_case(tokenizer, system, task):
    """Render native thinking-enabled prompt; return protocol case dict."""
    messages = [dict(role="system", content=system),
                dict(role="user", content=task["prompt"])]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not rendered.endswith(ASSISTANT_HEADER):
        raise ValueError("thinking-enabled render must end with bare assistant header: "
                         + task["id"])
    if OPEN in rendered or CLOSE in rendered:
        raise ValueError("thinking-enabled prompt must not contain think markers: " + task["id"])
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    answer = json.dumps(task.get("expected"), ensure_ascii=False)
    gold_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) + THINKING_MAX_TOKENS + 4 > CONTEXT:
        raise ValueError("prompt too long for 2048-token budget: " + task["id"])
    if len(gold_ids) > THINKING_MAX_TOKENS:
        raise ValueError("gold answer too long: " + task["id"])
    return {"id": task["id"], "rendered": rendered, "input_ids": list(ids),
            "answer": answer, "gold_ids": list(gold_ids)}


def assemble_suite(source_suite, tasks):
    """8-case suite cloning system/tasks; only limit + claim note change."""
    return {
        "schema": source_suite.get("schema", _base.SCHEMA),
        "seed": source_suite.get("seed"),
        "system": source_suite.get("system"),
        "quality_max_tokens": THINKING_MAX_TOKENS,
        "claim_limits": (source_suite.get("claim_limits", "")
                         + " Post-hoc exploratory thinking-enabled supplement"
                           " (8 of 16 -00 cases); not primary; no equivalence"
                           " or BF16-fidelity claim."),
        "tasks": copy.deepcopy(tasks),
    }


def assemble_protocol(original_protocol, suite_sha256, cases):
    """8-case protocol: thinking on, 2048 budget, v2 stops/cache/vocabs."""
    return {
        "suite_file_sha256": suite_sha256,
        "tokenizer_sha256": original_protocol["tokenizer_sha256"],
        "template_sha256": original_protocol["template_sha256"],
        "scope": ("Post-hoc exploratory thinking-enabled supplement: 8 procedural"
                  " validation cases (-00 subset), native <think>...</think> format,"
                  " 2048-token outputs; not primary, no equivalence or"
                  " BF16-fidelity claims."),
        "stop_ids": list(original_protocol["stop_ids"]),
        "context": original_protocol["context"],
        "cache": original_protocol["cache"],
        "thinking": True,
        "mtp": original_protocol["mtp"],
        "max_new_tokens": THINKING_MAX_TOKENS,
        "valid_vocabulary_size": original_protocol["valid_vocabulary_size"],
        "stored_vocabulary_size": original_protocol["stored_vocabulary_size"],
        "cases": cases,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _refuse_overwrite(path):
    if Path(path).exists():
        _fail(f"refusing to overwrite existing path: {path}")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def _cmd_build(args):
    _refuse_overwrite(args.output)
    suite_path, protocol_path, source = Path(args.suite), Path(args.protocol), Path(args.source)
    for path in (suite_path, protocol_path, source):
        if not path.exists():
            _fail(f"missing input path: {path}")
    suite_bytes = suite_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    suite = json.loads(suite_bytes.decode("utf-8"))
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if _sha256_bytes(suite_bytes) != protocol.get("suite_file_sha256"):
        _fail("suite file hash does not match original protocol")
    for name, key in (("tokenizer.json", "tokenizer_sha256"),
                      ("chat_template.jinja", "template_sha256")):
        blob = source / name
        if not blob.exists():
            _fail(f"missing source file: {blob}")
        if _sha256_bytes(blob.read_bytes()) != protocol.get(key):
            _fail(f"source {name} hash does not match original protocol")
    generation_config = source / "generation_config.json"
    if generation_config.exists():
        eos = json.loads(generation_config.read_text(encoding="utf-8")).get("eos_token_id")
        if eos is not None and list(eos) != list(protocol.get("stop_ids", [])):
            _fail("source stop ids do not match original protocol")
    frozen = (protocol.get("thinking") is False and protocol.get("max_new_tokens") == 128
              and protocol.get("context") == CONTEXT and protocol.get("cache") == CACHE
              and protocol.get("mtp") is False and len(protocol.get("cases", [])) == 64
              and suite.get("quality_max_tokens") == 128 and len(suite.get("tasks", [])) == 64)
    if not frozen:
        _fail("parent suite/protocol are not the frozen thinking-off v2 pair")
    from transformers import AutoTokenizer  # lazy: build only, registered env
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True,
                                              trust_remote_code=False)
    if len(tokenizer) != protocol.get("valid_vocabulary_size"):
        _fail("source tokenizer vocabulary does not match original protocol")
    try:
        selected = select_thinking_subset(suite.get("tasks", []))
        batch_a, batch_b = split_batches(selected)
        cases_a = [render_thinking_case(tokenizer, suite["system"], t) for t in batch_a]
        cases_b = [render_thinking_case(tokenizer, suite["system"], t) for t in batch_b]
    except ValueError as exc:
        _fail(str(exc))
    suites = {SUITE_A: assemble_suite(suite, batch_a), SUITE_B: assemble_suite(suite, batch_b)}
    blobs = {name: (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
             for name, obj in suites.items()}
    protocols = {
        PROTOCOL_A: assemble_protocol(protocol, _sha256_bytes(blobs[SUITE_A].encode("utf-8")),
                                      cases_a),
        PROTOCOL_B: assemble_protocol(protocol, _sha256_bytes(blobs[SUITE_B].encode("utf-8")),
                                      cases_b),
    }
    for name, obj in protocols.items():
        blobs[name] = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        Path(args.output).mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        _fail(f"refusing to overwrite existing path: {args.output}")
    out = Path(args.output)
    for name, blob in blobs.items():
        (out / name).write_text(blob, encoding="utf-8")
    manifest = {
        "parent_suite_sha256": _sha256_bytes(suite_bytes),
        "parent_protocol_sha256": _sha256_bytes(protocol_bytes),
        "tokenizer_sha256": protocol["tokenizer_sha256"],
        "template_sha256": protocol["template_sha256"],
        "thinking": True, "max_new_tokens": THINKING_MAX_TOKENS,
        "context": CONTEXT, "cache": CACHE, "mtp": False,
        "batches": {"a": [t["id"] for t in batch_a], "b": [t["id"] for t in batch_b]},
        "files": {name: _sha256_bytes(blob.encode("utf-8")) for name, blob in blobs.items()},
        "note": "Post-hoc exploratory supplement; frozen v2 primary unchanged.",
    }
    _write_json(out / MANIFEST, manifest)
    max_prompt = max(len(c["input_ids"]) for c in cases_a + cases_b)
    print(json.dumps({"batches": manifest["batches"], "max_prompt_ids": max_prompt,
                      "files": manifest["files"]}, indent=2, ensure_ascii=False))


def _cmd_score(args):
    _refuse_overwrite(args.output)
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    results_obj = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = score_suite_thinking(suite, results_obj)
    _write_json(args.output, report)
    totals = report["totals"]
    strip_errors = sum(1 for c in report["cases"] if c["strip_error"] is not None)
    print(f"passed {totals['passed']}/{totals['total']} "
          f"rate={totals['rate']:.3f} "
          f"wilson=[{totals['wilson_low']:.3f},{totals['wilson_high']:.3f}] "
          f"thinking_parse_errors={strip_errors}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build two 8-case thinking-enabled suites+protocols.")
    build.add_argument("--suite", required=True, help="Original frozen v2 suite file.")
    build.add_argument("--protocol", required=True, help="Original frozen v2 protocol file.")
    build.add_argument("--source", required=True, help="Local original model directory.")
    build.add_argument("--output", required=True, help="New output directory (must not exist).")
    score = sub.add_parser("score", help="Score results with think-block pre-strip.")
    score.add_argument("--suite", required=True)
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        _cmd_build(args)
    elif args.command == "score":
        _cmd_score(args)


if __name__ == "__main__":
    main()
