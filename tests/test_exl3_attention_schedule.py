"""Host policy tests; GPU numerical validation is a separate requirement."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'vendor/rocm-exl3/exllamav3/modules/attention_fn'
spec=importlib.util.spec_from_file_location('attention_schedule',DIRECTORY/'schedule.py')
schedule=importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class AttentionScheduleTests(unittest.TestCase):
    def pick(self, **changes):
        fields=dict(arch='gfx1101',batch=1,query_rows=1,query_heads=16,kv_heads=4,
                    head_dim=256,occupied_bound=32768,k_bits=8,v_bits=8)
        return schedule.decode_options(**(fields|changes))

    def test_formats_and_geometries_have_measured_paths(self):
        for heads in (16,24):
            for bits in (0,4,5,6,8):
                for rows in (1,2,3,4):
                    options=self.pick(query_heads=heads,k_bits=bits,v_bits=bits,query_rows=rows)
                    self.assertTrue(options['parallel_combine'])
                    self.assertGreater(options['num_splits'],0)
                    if 'head_block' in options:
                        self.assertLessEqual((1<<(rows-1).bit_length())*options['head_block'],64)

    def test_unsupported_cases_keep_fallback(self):
        for change in (dict(arch='gfx1201'),dict(batch=2),dict(query_rows=0),dict(query_rows=9),
                       dict(head_dim=128),dict(query_heads=32),dict(kv_heads=8),
                       dict(occupied_bound=16383),dict(occupied_bound=131073),
                       dict(k_bits=3,v_bits=3),dict(k_bits=4,v_bits=8),
                       dict(query_rows=5,k_bits=4,v_bits=4)):
            with self.subTest(change=change): self.assertEqual(self.pick(**change),{})

    def test_short_and_long_groups_are_distinct(self):
        self.assertEqual(self.pick(occupied_bound=8192),{})
        short=self.pick(occupied_bound=16384)
        long=self.pick(occupied_bound=32768)
        self.assertEqual((short['block_n'],short['num_splits']),(32,32))
        self.assertEqual((long['block_n'],long['num_splits']),(16,64))
        self.assertNotIn('head_block',self.pick(query_rows=3))
        self.assertEqual(self.pick(query_heads=24,query_rows=3)['head_block'],8)
        self.assertEqual(self.pick(query_heads=24,query_rows=3,k_bits=5,v_bits=5)['head_block'],8)

    def test_format_changes_tile_not_precision(self):
        for bits in (5,6):
            options=self.pick(k_bits=bits,v_bits=bits)
            self.assertEqual(options['block_n'],16)
            self.assertNotIn('k_bits',options)
            self.assertNotIn('v_bits',options)

    def test_larger_verify_batches_fit_the_kernel_row_limit(self):
        for heads in (16,24):
            for bits in (0,5,6,8):
                for rows in (5,6,7,8):
                    options=self.pick(query_heads=heads,k_bits=bits,v_bits=bits,query_rows=rows)
                    block_h=options.get('head_block',2)
                    self.assertLessEqual(8*block_h,64)
                    self.assertGreaterEqual(8*block_h,16)

    def test_additional_geometries_are_selected_without_model_identity(self):
        for heads,kv,dim in ((32,8,128),(16,2,128),(32,4,128),(32,8,256)):
            for bits in (0,8):
                options=self.pick(query_heads=heads,kv_heads=kv,head_dim=dim,
                                  query_rows=3,k_bits=bits,v_bits=bits)
                self.assertTrue(options['parallel_combine'])
                self.assertEqual(self.pick(query_heads=heads,kv_heads=kv,head_dim=dim,
                                          occupied_bound=65537),{})

    def test_mixed_cache_format_is_never_rewritten(self):
        for k,v in ((8,4),(6,4),(8,6),(5,8)):
            options=self.pick(k_bits=k,v_bits=v,query_heads=24,query_rows=3)
            self.assertEqual(options['head_block'],8)
            self.assertEqual(set(options),{'block_n','num_splits','parallel_combine','head_block'})


class AttentionAdapterTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((DIRECTORY/'triton_paged.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_auto_decode_options')
        # Inject the pure policy; importing Torch/Triton would defeat host-only tests.
        fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
        self.props=Mock(return_value=NS(gcnArchName='gfx1101:sramecc-'))
        self.env=dict(_qc_decode_profile='auto',_is_rocm=True,_qc_device_arch={},
                      torch=NS(cuda=NS(get_device_properties=self.props)),decode_options=schedule.decode_options)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'policy','exec'),self.env)
        self.policy=self.env[fn.name]

    def args(self,**changes):
        cache=NS(shape=(256,256,128))
        base=dict(bsz=1,q_len=1,num_q_heads=16,num_kv_heads=4,dim=256,
            causal=True,cu_seqlens=None,sinks=None,window_size=None,softcap=0.,non_causal_spans=None,
            block_table=NS(shape=(1,128)),q_cache=(cache,None,cache,None,8,8),k_cache=None,
            q=NS(device='cuda:0'),cache_seqlens=Mock())
        return NS(**(base|changes))

    def test_table_bound_and_physical_cap_not_capacity(self):
        a=self.args(block_table=NS(shape=(1,16)))
        self.assertEqual(self.policy(a),{})
        cache=NS(shape=(16,256,128))
        a=self.args(q_cache=(cache,None,cache,None,8,8))
        self.assertEqual(self.policy(a),{})
        a=self.args(block_table=NS(shape=(1,64)))
        self.assertEqual(self.policy(a)['num_splits'],32)
        a.cache_seqlens.assert_not_called()

    def test_device_info_is_cached(self):
        self.policy(self.args());self.policy(self.args())
        self.props.assert_called_once_with('cuda:0')

    def test_other_semantics_and_codecs_are_unchanged(self):
        for change in (dict(causal=False),dict(sinks=object()),dict(window_size=128),
                       dict(softcap=1.),dict(cu_seqlens=object()),dict(non_causal_spans=[(0,1,True)]),
                       dict(q_cache=(*self.args().q_cache,object())),dict(block_table=None)):
            self.assertEqual(self.policy(self.args(**change)),{})
        self.props.assert_not_called()

    def test_explicit_old_profiles_remain_old(self):
        for name in ('default','long'):
            self.env['_qc_decode_profile']=name
            self.assertEqual(self.policy(self.args()),{})

    def test_f16_uses_same_metadata_policy(self):
        a=self.args(q_cache=None,k_cache=NS(shape=(128,256,4,256)))
        self.assertEqual(self.policy(a)['block_n'],16)


if __name__=='__main__': unittest.main()
