"""CPU stub check of the actual compact quantizer entry, stopping at upload."""

import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    function = next(n for n in ast.parse(args.source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "quantize_exl3")
    # Postpone source annotations; dependencies past the upload are intentionally absent.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              function], type_ignores=[])
    events = []
    fake_torch = SimpleNamespace(
        float="float", half="half", bfloat16="bfloat16", device=lambda d: d,
        manual_seed=lambda s: events.append("seed"),
        cuda=SimpleNamespace(synchronize=lambda d: events.append("synchronize:" + d),
                             empty_cache=lambda: events.append("empty_cache")),
    )
    class StopAtUpload(Exception):
        pass

    class Weight:
        shape = (5120, 248320)
        device = "cpu"
        dtype = "float"

        def __init__(self, count):
            self.count = count

        def numel(self):
            return self.count

        def to(self, device):
            events.append("upload:" + device)
            raise StopAtUpload

    scope = {"torch": fake_torch, "ProgressBar": lambda *a: nullcontext(),
             "get_temp_buffers": SimpleNamespace(cache_clear=lambda: events.append("clear_scratch")),
             "report_quant_memory": lambda *a: None}
    exec(compile(ast.fix_missing_locations(module), str(args.source), "exec"), scope)
    results = []
    for compact, count, expected in (
        (True, 1271398400, ["synchronize:cuda:0", "clear_scratch", "empty_cache", "upload:cuda:0"]),
        (True, 32768, ["upload:cuda:0"]),
        (False, 1271398400, ["upload:cuda:0"]),
    ):
        events.clear()
        try:
            scope["quantize_exl3"](Weight(count), {}, {"devices": ["cuda:0"], "compact_cpu_buffers": compact}, False)
        except StopAtUpload:
            pass
        assert events == expected, events
        results.append(dict(compact=compact, elements=count, events=list(events)))
    result = dict(cases=results, gpu_executed=False, actual_tensors_allocated=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
