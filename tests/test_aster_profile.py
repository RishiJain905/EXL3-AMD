"""CPU checks for the metadata-only Aster scheduling boundary, not GPU correctness."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/rocm-exl3/exllamav3/modules/attention_fn/triton_paged.py"


class AsterProfileTests(unittest.TestCase):
    def setUp(self):
        # Import just the policy so host CLI tests require neither Torch nor Triton.
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_aster_qc_options")
        self.props = Mock(return_value=NS(gcnArchName="gfx1101:sramecc-"))
        self.env = {"_qc_decode_profile": "long", "_is_rocm": True,
                    "_qc_device_arch": {},
                    "torch": NS(cuda=NS(get_device_properties=self.props))}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), self.env)
        self.policy = self.env[fn.name]

    def args(self, **overrides):
        cache = NS(shape=(469, 256, 160))
        fields = dict(bsz=1, dim=256, num_q_heads=24, num_kv_heads=4,
                      q_cache=(cache, None, cache, None, 5, 5, object(), (0.83, 0.17)),
                      q=NS(device="cuda:0"), block_table=NS(shape=(1, 128)),
                      q_len=7, causal=True, cu_seqlens=None, sinks=None,
                      window_size=None, softcap=0.0, non_causal_spans=None)
        return NS(**(fields | overrides))

    def test_opt_in_and_supported_geometry(self):
        self.assertTrue(self.policy(self.args()))
        self.assertTrue(self.policy(self.args(q_len=1024), prefill=True))
        self.env["_qc_decode_profile"] = "default"
        self.assertEqual(self.policy(self.args()), {})

    def test_other_codecs_and_lookup_only_are_unchanged(self):
        for bits, poly, arity in ((8, None, 6), (5, None, 6), (5, None, 8)):
            cache = list(self.args().q_cache)
            cache[4:6], cache[7] = [bits, bits], poly
            self.assertEqual(self.policy(self.args(q_cache=tuple(cache[:arity]))), {})
        self.props.assert_not_called()

    def test_unmeasured_devices_and_shapes_decline(self):
        for changed in (dict(bsz=2), dict(dim=128), dict(num_q_heads=32),
                        dict(num_kv_heads=8), dict(causal=False), dict(sinks=object()),
                        dict(window_size=128), dict(softcap=1.0),
                        dict(cu_seqlens=object()), dict(non_causal_spans=[(0, 1, True)])):
            with self.subTest(changed=changed):
                self.assertEqual(self.policy(self.args(**changed)), {})
        self.props.assert_not_called()
        self.props.return_value = NS(gcnArchName="gfx1201")
        self.assertEqual(self.policy(self.args()), {})

    def test_context_bounds_use_physical_page_cap(self):
        for tokens in (3840, 120320):
            args = self.args(block_table=NS(shape=(1, tokens // 256)))
            if tokens > 120064:
                args.q_cache = (NS(shape=(tokens // 256, 256, 160)), *args.q_cache[1:])
            self.assertEqual(self.policy(args), {})
        for tokens in (4096, 16384, 32768, 120064):
            self.assertTrue(self.policy(self.args(block_table=NS(shape=(1, tokens // 256)))))
        self.assertTrue(self.policy(self.args(block_table=NS(shape=(1, 480)))))

    def test_query_limits(self):
        for qlen in range(1, 9):
            self.assertTrue(self.policy(self.args(q_len=qlen)))
        for qlen in (0, 9, 16):
            self.assertEqual(self.policy(self.args(q_len=qlen)), {})
        for qlen in (17, 255, 1025):
            self.assertEqual(self.policy(self.args(q_len=qlen), prefill=True), {})
        for qlen in (256, 1024):
            self.assertTrue(self.policy(self.args(q_len=qlen), prefill=True))

    def test_long_schedule_stays_within_measured_query_groups(self):
        def options(tokens, qlen):
            return self.policy(self.args(q_len=qlen,
                                         block_table=NS(shape=(1, tokens // 256))))
        for tokens in (65536, 110080, 120064):
            self.assertEqual(options(tokens, 1)["num_splits"], 128)
            for qlen in (5, 6, 7, 8):
                tuned = options(tokens, qlen)
                self.assertEqual((tuned["num_splits"], tuned["head_block"],
                                  tuned["num_warps"]), (128, 4, 4))
            for qlen in (2, 3, 4):
                self.assertEqual(options(tokens, qlen), options(65280, qlen))
        self.assertEqual(options(65280, 1)["num_splits"], 32)
        self.assertEqual(options(65280, 7)["num_splits"], 64)


if __name__ == "__main__":
    unittest.main()
