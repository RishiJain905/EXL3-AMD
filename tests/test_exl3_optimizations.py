"""The speculative shortcut must decline stateful or customized sampling."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from quantlab.methods.exl3.optimizations import can_batch_greedy, configure_native


class NativeOptionsTests(unittest.TestCase):
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
            self.assertEqual(configure_native('unused')['smallm_kernel'],'dot')
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
