"""The speculative shortcut must decline stateful or customized sampling."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from quantlab.methods.exl3.optimizations import can_batch_greedy, configure_native


class NativeOptionsTests(unittest.TestCase):
    def test_packed_head_uses_qualified_abi_and_preserves_older_binaries(self):
        import os
        for version, enabled in ((1, False), (2, False), (3, True)):
            def capability(): return version
            library = SimpleNamespace(quantlab_exl3_head_repacked_abi=capability)
            with patch('ctypes.CDLL', return_value=library), patch.dict(os.environ, {}, clear=True):
                result = configure_native('verified.so', native_smallm=True)
                self.assertEqual(result['head_repacked_abi'], version)
                self.assertIs(result['head_repacked'], enabled)
        with patch('ctypes.CDLL', return_value=object()):
            result = configure_native('legacy.so', native_smallm=True)
            self.assertIsNone(result['head_repacked_abi'])
            self.assertFalse(result['head_repacked'])

    def test_packed_head_unknown_abi_fails_closed(self):
        def capability(): return 7
        with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_head_repacked_abi=capability)):
            with self.assertRaisesRegex(ValueError, 'packed head ABI'):
                configure_native('unknown.so', native_smallm=True)

    def test_packed_head_preserves_explicit_head_and_wmma_controls(self):
        import os
        def capability(): return 3
        def optimization(): return 2
        library = SimpleNamespace(quantlab_exl3_head_repacked_abi=capability,
                                  quantlab_exl3_optimization_abi=optimization)
        with patch('ctypes.CDLL', return_value=library), patch.dict(os.environ, {}, clear=True):
            for options in ({'head_warps': 1}, {'smallm_kernel': 'wmma'}, {'smallm_kernel': 'wmma-register'}):
                self.assertFalse(configure_native('verified.so', native_smallm=True, **options)['head_repacked'])

    def test_auto_prefill_selects_verified_wmma_with_legacy_fallback(self):
        import os
        def version(): return 1
        with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_hgemm_abi=version)), \
                patch.dict(os.environ, {}, clear=True):
            result = configure_native('verified.so', native_smallm=True)
            self.assertEqual(result['prefill_gemm'], 'wmma')
            self.assertEqual(os.environ['EXL3_HGEMM_IMPL'], 'wmma')
            self.assertEqual(configure_native('verified.so', native_smallm=True, prefill_gemm='blas')['prefill_gemm'], 'blas')
        with patch('ctypes.CDLL', return_value=object()):
            self.assertEqual(configure_native('legacy.so', native_smallm=True)['prefill_gemm'], 'blas')
            with self.assertRaisesRegex(ValueError, 'prefill GEMM'):
                configure_native('legacy.so', prefill_gemm='wmma')
        def bad(): return 99
        with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_hgemm_abi=bad)):
            with self.assertRaisesRegex(ValueError, 'ABI'):
                configure_native('unknown.so', native_smallm=True)
            self.assertEqual(configure_native('unknown.so', native_smallm=True, prefill_gemm='blas')['prefill_gemm'], 'blas')

    def test_prefill_pair_capabilities_and_fallback(self):
        import os
        for name in ('packed_prefill', 'mlp_pair'):
            def version(): return 1
            symbol = 'quantlab_exl3_' + name + '_abi'
            for requested in (None, True, False):
                with patch('ctypes.CDLL', return_value=SimpleNamespace(**{symbol: version})), \
                        patch.dict(os.environ, {}, clear=True):
                    result = configure_native('verified.so', native_smallm=True, **{name: requested})
                    self.assertIs(result[name], requested is not False)
                    self.assertEqual(result[name + '_abi'], None if requested is False else 1)
            with patch('ctypes.CDLL', return_value=object()):
                self.assertFalse(configure_native('legacy.so', native_smallm=True, **{name: None})[name])
                with self.assertRaisesRegex(ValueError, 'ABI 1'):
                    configure_native('legacy.so', native_smallm=True, **{name: True})
            def bad(): return 99
            with patch('ctypes.CDLL', return_value=SimpleNamespace(**{symbol: bad})):
                with self.assertRaisesRegex(ValueError, 'Unsupported'):
                    configure_native('unknown.so', native_smallm=True, **{name: None})
                self.assertFalse(configure_native('unknown.so', native_smallm=True, **{name: False})[name])
            with patch('ctypes.CDLL') as library:
                for value in ('yes', 1, True):
                    with self.assertRaises(ValueError):
                        configure_native('unused', **{name: value})
                library.assert_not_called()

    def test_mlp_pair_preserves_explicit_wmma_diagnostic(self):
        import os
        def version(): return 1
        library = SimpleNamespace(quantlab_exl3_optimization_abi=version, quantlab_exl3_mlp_pair_abi=version)
        with patch('ctypes.CDLL', return_value=library), patch.dict(os.environ, {}, clear=True):
            result = configure_native('verified.so', native_smallm=True, smallm_kernel='wmma')
            self.assertFalse(result['mlp_pair'])
            self.assertEqual(os.environ['EXL3_SMALLM_WMMA'], '1')
            with self.assertRaisesRegex(ValueError, 'dot'):
                configure_native('verified.so', native_smallm=True, smallm_kernel='wmma', mlp_pair=True)

    def _new_library(self, packed_abi=1, highbit_abi=1, codebooks=5):
        def mid(): return packed_abi
        def high(): return highbit_abi
        def books(): return codebooks
        return SimpleNamespace(quantlab_exl3_packed_mid_abi=mid,
                              quantlab_exl3_smallm_highbit_abi=high,
                              quantlab_exl3_smallm_codebooks=books)

    def test_packed_options_require_verified_capability(self):
        with patch('ctypes.CDLL', return_value=object()):
            for options in ({'packed_mid': True}, {'mlp_warps': 4}):
                with self.assertRaisesRegex(ValueError, 'packed-mid ABI'):
                    configure_native('old.so', native_smallm=True, **options)

    def test_packed_auto_enables_on_new_binary(self):
        import os
        with patch('ctypes.CDLL', return_value=self._new_library()), patch.dict(os.environ, {}, clear=True):
            result = configure_native('new.so', native_smallm=True)
            self.assertIs(result['packed_mid'], True)
            self.assertEqual(result['packed_mid_abi'], 1)
            self.assertIsNone(result['mlp_warps'])
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '1')
            self.assertNotIn('EXL3_SMALLM_MLP_WARPS', os.environ)

    def test_packed_auto_falls_back_on_legacy_binary(self):
        import os
        with patch('ctypes.CDLL', return_value=object()), patch.dict(os.environ, {}, clear=True):
            result = configure_native('old.so', native_smallm=True)
            self.assertIs(result['packed_mid'], False)
            self.assertIsNone(result['packed_mid_abi'])
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '0')

    def test_packed_auto_without_smallm_stays_off_without_loading(self):
        import os
        with patch('ctypes.CDLL') as library, patch.dict(os.environ, {}, clear=True):
            result = configure_native('unused')
            self.assertIs(result['packed_mid'], False)
            self.assertIsNone(result['packed_mid_abi'])
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '0')
            library.assert_not_called()

    def test_packed_explicit_off_disables_without_probing(self):
        import os
        def bad(): return 99
        cases = [self._new_library(), object(),
                 SimpleNamespace(quantlab_exl3_packed_mid_abi=bad)]
        for library in cases:
            with self.subTest(library=type(library).__name__):
                with patch('ctypes.CDLL', return_value=library), patch.dict(os.environ, {}, clear=True):
                    result = configure_native('any.so', native_smallm=True, packed_mid=False)
                    self.assertIs(result['packed_mid'], False)
                    self.assertEqual(os.environ['EXL3_PACKED_MID'], '0')

    def test_packed_capabilities_and_environment_reset(self):
        import os
        library = self._new_library()
        with patch('ctypes.CDLL', return_value=library), patch.dict(os.environ, {}, clear=True):
            result = configure_native('new.so', native_smallm=True, packed_mid=True, mlp_warps=4)
            self.assertEqual(result['smallm_highbit_abi'], 1)
            self.assertEqual(result['packed_mid_abi'], 1)
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '1')
            self.assertEqual(os.environ['EXL3_SMALLM_MLP_WARPS'], '4')
            # Automatic selection keeps the new binary enabled but clears a stale override.
            result = configure_native('new.so', native_smallm=True)
            self.assertIs(result['packed_mid'], True)
            self.assertIsNone(result['mlp_warps'])
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '1')
            self.assertNotIn('EXL3_SMALLM_MLP_WARPS', os.environ)
        with patch('ctypes.CDLL', return_value=object()), patch.dict(os.environ, {'EXL3_PACKED_MID': '1', 'EXL3_SMALLM_MLP_WARPS': '8'}):
            result = configure_native('old.so', native_smallm=True)
            self.assertIs(result['packed_mid'], False)
            self.assertEqual(os.environ['EXL3_PACKED_MID'], '0')
            self.assertNotIn('EXL3_SMALLM_MLP_WARPS', os.environ)

    def test_mlp_warps_requires_verified_capability_when_auto(self):
        with patch('ctypes.CDLL', return_value=object()):
            with self.assertRaisesRegex(ValueError, 'packed-mid ABI'):
                configure_native('old.so', native_smallm=True, mlp_warps=4)

    def test_packed_options_fail_before_loading_when_invalid(self):
        with patch('ctypes.CDLL') as library:
            for options in ({'packed_mid': True}, {'mlp_warps': 8},
                            {'packed_mid': 'yes'}, {'mlp_warps': 2},
                            {'packed_mid': 1}, {'mlp_warps': 4, 'native_smallm': False}):
                with self.assertRaises(ValueError):
                    configure_native('unused', **options)
            library.assert_not_called()

    def test_unknown_packed_abi_rejects(self):
        def bad(): return 99
        for symbol in ('quantlab_exl3_packed_mid_abi', 'quantlab_exl3_smallm_highbit_abi'):
            with patch('ctypes.CDLL', return_value=SimpleNamespace(**{symbol: bad})):
                with self.assertRaisesRegex(ValueError, 'ABI'):
                    configure_native('new.so', native_smallm=True, packed_mid=True)

    def test_packed_auto_unknown_abi_rejects_when_probed(self):
        def bad(): return 99
        with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_packed_mid_abi=bad)):
            with self.assertRaisesRegex(ValueError, 'ABI'):
                configure_native('new.so', native_smallm=True)

    def test_highbit_capability_requires_exact_supported_abi(self):
        for abi in (1, 2):
            def version(): return abi
            lib = SimpleNamespace(quantlab_exl3_smallm_highbit_abi=version)
            with patch('ctypes.CDLL', return_value=lib):
                if abi == 1:
                    self.assertEqual(configure_native('new.so', native_smallm=True)['smallm_highbit_abi'], 1)
                else:
                    with self.assertRaisesRegex(ValueError, 'high-bit ABI'):
                        configure_native('new.so', native_smallm=True)

    def test_old_binary_does_not_enable_highbit(self):
        with patch('ctypes.CDLL', return_value=object()):
            self.assertIsNone(configure_native('old.so', native_smallm=True)['smallm_highbit_abi'])

    def test_defaults_do_not_require_new_binary(self):
        with patch('ctypes.CDLL') as library, patch.dict('os.environ',{},clear=True):
            result = configure_native('unused')
            self.assertEqual(result['smallm_kernel'],'dot')
            self.assertIs(result['packed_mid'], False)
            library.assert_not_called()

    def test_old_binary_rejects_explicit_new_kernel(self):
        with patch('ctypes.CDLL',return_value=object()):
            with self.assertRaisesRegex(ValueError,'newer'):
                configure_native('old.so',smallm_kernel='wmma')

    def test_unknown_abi_rejects(self):
        def version(): return 42
        with patch('ctypes.CDLL',return_value=SimpleNamespace(quantlab_exl3_optimization_abi=version)):
            with self.assertRaisesRegex(ValueError,'ABI'):
                configure_native('new.so',head_warps=4)

    def test_invalid_options_do_not_load_native_code(self):
        with patch('ctypes.CDLL') as library:
            for kwargs in ({'smallm_kernel':'unknown'},{'head_warps':3}):
                with self.assertRaises(ValueError):configure_native('unused',**kwargs)
            library.assert_not_called()


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        class SS_Argmax:
            pass
        class SS_Fused:
            MODE_GREEDY = 0
        class ArgmaxSampler:
            def __init__(self):
                self.steps = [SS_Argmax()]
                self.reqs_past_ids = self.reqs_torch_seed = False
        module = ModuleType('exllamav3.generator.sampler')
        module.ArgmaxSampler = ArgmaxSampler
        custom = ModuleType('exllamav3.generator.sampler.custom')
        custom.SS_Argmax, custom.SS_Fused = SS_Argmax, SS_Fused
        self.patch = patch.dict(sys.modules, {'exllamav3.generator.sampler': module,
                                             'exllamav3.generator.sampler.custom': custom})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.job = SimpleNamespace(sampler=ArgmaxSampler(), new_tokens=0,
            min_new_tokens=0, filters=[], banned_strings=[], forced_ids=None,
            device_logit_mask=None, return_probs=False, return_top_tokens=0)
        self.logits = SimpleNamespace(shape=(1,5,100))

    def test_plain_greedy_is_eligible(self):
        self.assertTrue(can_batch_greedy(self.job,self.logits))

    def test_stateful_features_decline(self):
        for key,value in dict(new_tokens=-1,min_new_tokens=1,filters=[object()],
            banned_strings=['bad'],forced_ids=object(),device_logit_mask=object(),
            return_probs=True,return_top_tokens=1).items():
            original=getattr(self.job,key)
            setattr(self.job,key,value)
            self.assertFalse(can_batch_greedy(self.job,self.logits),key)
            setattr(self.job,key,original)

    def test_instrumented_sampler_and_subclasses_decline(self):
        self.job.sampler.forward=lambda *args: None
        self.assertFalse(can_batch_greedy(self.job,self.logits))
        parent=type(self.job.sampler)
        self.job.sampler=type('CustomArgmax',(parent,),{})()
        self.assertFalse(can_batch_greedy(self.job,self.logits))
        self.job.sampler=object()
        self.assertFalse(can_batch_greedy(self.job,self.logits))

    def test_single_position_or_cfg_declines(self):
        for shape in ((1,1,100),(2,5,100)):
            self.assertFalse(can_batch_greedy(self.job,SimpleNamespace(shape=shape)))

    def test_mutated_sampler_stack_and_state_requirements_decline(self):
        for name in ('reqs_past_ids','reqs_torch_seed'):
            setattr(self.job.sampler,name,True)
            self.assertFalse(can_batch_greedy(self.job,self.logits))
            setattr(self.job.sampler,name,False)
        self.job.sampler.steps.insert(0,object())
        self.assertFalse(can_batch_greedy(self.job,self.logits))
