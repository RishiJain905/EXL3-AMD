# MiMo held-out validation fixtures

Small procedural validation suite for MiMo step 2. It is not a public
benchmark, not an equivalence test, and not a BF16 fidelity claim. It makes
no code-execution or real tool-use success claim: model output is scored as
JSON text only, generated code is never executed, and fictional tools are
never called.

## Scope

- 64 deterministic tasks, seed `20260922`, stdlib only.
- 16 tasks per category (`code`, `reasoning`, `tools`, `general`).
- Four families per category with four parameterizations each.
- Response budget: `quality_max_tokens: 128`.
- System prompt: `Follow the user instructions. Return only the requested JSON value.`
- No calibration data is used or embedded. Calibration sources are separate
  public datasets owned by the parent step.

Families:

- `code_filter_reduce`, `code_group_count`, `code_loop_trace`,
  `code_dict_state`: predict the exact JSON result of a small explicit
  Python program.
- `reasoning_inventory`, `reasoning_ratio`, `reasoning_order`,
  `reasoning_schedule`: integer inventory arithmetic, integer-ratio answers,
  bounded ordering constraints, and earliest-completion schedules under
  unlimited parallelism.
- `tools_choose_single`, `tools_choose_multi`: choose one fictional tool and
  arguments from tools listed in the prompt; expected JSON is
  `{"name": ..., "arguments": {...}}`.
- `tools_normalize`, `tools_extract`: normalize or extract a small JSON
  object under explicit rules.
- `general_lookup`, `general_sort`, `general_filter`, `general_aggregate`:
  grounded extraction, sorting, filtering, and counting over invented facts
  included in each prompt. No external world-knowledge grading.

## Use

Generate once; the parent run freezes the suite hash before full conversion:

```powershell
python scripts/mimo_validation_suite.py create --output .runtime/mimo-suite.json
python scripts/evaluate_exl3_candidate.py --mode quality --suite .runtime/mimo-suite.json ...
python scripts/mimo_validation_suite.py score --suite .runtime/mimo-suite.json --results <result.json> --output .runtime/mimo-report.json
```

Both commands refuse to overwrite an existing output file.

The scorer accepts:

- `evaluate_exl3_candidate.py` `result.json` (`{"results": [{"name", "output_text", ...}]}`),
- a single case file (`{"name", "output_text"}`),
- a simple mapping (`{task_id: response_text}`),
- or a list of `{"task_id", "response"}` entries.

## Scoring rules

- At most one optional fenced JSON block. Multiple fences, prose around the
  fence, trailing junk, empty output, non-string output, and non-JSON
  constants are parse failures.
- Exact recursive comparison: object key order is irrelevant, array order is
  relevant, booleans never equal numbers, and integers may equal integral
  floats.
- Missing and duplicate results fail explicitly. Unknown result IDs are
  reported separately. Totals always use all 64 suite tasks as the
  denominator; results are never quietly dropped.
- Reports include per-case pass/parse/scoring status, grouped totals by
  category and family, and Wilson 95% confidence intervals.

## Limits and step 3 boundary

This suite checks narrow procedural behaviors only. It does not establish
general coding ability, reasoning quality, production tool-use reliability,
retrieval quality, serving performance, or closeness to BF16.

Final selection testing belongs to step 3 and is intentionally untouched
here. Do not generate or run it from this package. Its families differ, for
example code synthesis with unit tests, graph search, multi-turn tool
recovery, and long-context grounded tasks.
