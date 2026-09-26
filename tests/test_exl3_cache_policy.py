"""Offline policy guards and per-component constructor plumbing."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

from quantlab.methods.exl3.cache_policy import load_cache_policy, validate_cache_policy, policy_summary, MAX_BYTES
from quantlab.methods.exl3.cache_precision import model_cache_options, resolve_cache_types, add_cache_precision_args


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'policy.json'
        self.config = b'{"model_type":"test"}'
        self.layer = dict(layer_idx=3, kv_heads=4, head_dim=256, k='aster5', v='aster5')
        self.policy = dict(version=1, config_sha256=hashlib.sha256(self.config).hexdigest(),
                           components=dict(target=[self.layer]))

    def write(self, value=None):
        self.path.write_text(json.dumps(self.policy if value is None else value), encoding='utf-8')
        return self.path

    def descriptors(self):
        return dict(target=[{k: self.layer[k] for k in ('layer_idx', 'kv_heads', 'head_dim')}])

    def test_valid_profile_and_summary(self):
        policy = load_cache_policy(self.write())
        self.assertEqual(validate_cache_policy(policy, self.config, self.descriptors()), {'target': {3: ('aster5', 'aster5')}})
        summary = policy_summary(policy)
        self.assertNotIn(str(self.path), json.dumps(summary))
        self.assertEqual(summary['sha256'], hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest())

    def test_bad_schema(self):
        for key, value in [('version', True), ('version', 2), ('config_sha256', 'z'*64), ('extra', 'unexpected')]:
            policy = copy.deepcopy(self.policy)
            policy[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                load_cache_policy(self.write(policy))
        for key, value in [('layer_idx', True), ('layer_idx', -1), ('head_dim', 0), ('kv_heads', 65537),
                           ('k', 'f16'), ('k', 'q8'), ('v', []), ('extra', 1)]:
            policy = copy.deepcopy(self.policy)
            policy['components']['target'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                load_cache_policy(self.write(policy))

    def test_duplicate_keys_layers_and_oversize(self):
        raw = json.dumps(self.policy)
        for content in ['{"version":1,"version":1}', raw.replace('"version": 1', '"version": NaN'), ' '* (MAX_BYTES+1), '{']:
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                load_cache_policy(self.path)
        self.policy['components']['target'].append(dict(self.layer))
        with self.assertRaises(ValueError):
            load_cache_policy(self.write())

    def test_binding_coverage_and_draft(self):
        for config, descriptors in [(b'wrong', self.descriptors()), (self.config, {'target': []}),
                                    (self.config, dict(target=[dict(layer_idx=4, kv_heads=4, head_dim=256)])),
                                    (self.config, dict(target=[dict(layer_idx=3, kv_heads=8, head_dim=256)])),
                                    (self.config, dict(self.descriptors(), draft=[]))]:
            with self.assertRaises(ValueError):
                validate_cache_policy(self.policy, config, descriptors)
        self.policy['components']['draft'] = [dict(self.layer, layer_idx=64, k='q8', v='q6')]
        self.assertEqual(set(validate_cache_policy(self.policy, self.config, self.descriptors())), {'target'})
        descriptors = dict(self.descriptors(), draft=[dict(layer_idx=64, kv_heads=4, head_dim=256)])
        self.assertEqual(validate_cache_policy(self.policy, self.config, descriptors)['draft'][64], ('q8', 'q6'))

    def test_all_components_validated_before_constructor_options(self):
        self.policy['components']['draft'] = [dict(self.layer, layer_idx=64, k='q8', v='q6')]
        path = self.write()
        args = add_cache_precision_args(argparse.ArgumentParser()).parse_args(['--cache-type', 'aster5', '--cache-policy', str(path)])
        class Model:
            def __init__(self, idx):
                self.idx = idx
            def get_cache_layers(self):
                return [SimpleNamespace(layer_idx=self.idx, num_kv_heads=4, head_dim=256)]
        quant, aster = type('Quant', (), {}), type('Aster', (), {})
        options, summary = model_cache_options(args, self.config, dict(target=Model(3), draft=Model(64)),
                                              quant_layer_type=quant, aster_layer_type=aster)
        self.assertIs(options['target']['layer_overrides'][3]['layer_type'], aster)
        self.assertEqual(options['draft']['layer_overrides'][64], dict(layer_type=quant, k_bits=8, v_bits=6, compand_a=0))
        self.assertEqual(summary['version'], 1)
        with self.assertRaises(ValueError):
            model_cache_options(args, self.config, dict(target=Model(3), draft=Model(65)),
                                quant_layer_type=quant, aster_layer_type=aster)

    def test_f16_policy_cannot_bypass_quantized_guards(self):
        args = add_cache_precision_args(argparse.ArgumentParser()).parse_args(['--cache-policy', str(self.write())])
        with self.assertRaisesRegex(ValueError, 'quantized'):
            resolve_cache_types(args)

    def test_target_and_draft_may_reuse_layer_ids_without_sharing_precision(self):
        self.policy['components']['draft'] = [dict(self.layer, k='q8', v='q6')]
        descriptors = self.descriptors()
        descriptors['draft'] = copy.deepcopy(descriptors['target'])
        assignments = validate_cache_policy(self.policy, self.config, descriptors)
        self.assertEqual(assignments['target'][3], ('aster5', 'aster5'))
        self.assertEqual(assignments['draft'][3], ('q8', 'q6'))

    def test_host_filename_import_before_src_is_available(self):
        source = Path(__file__).resolve().parents[1] / 'src/quantlab/methods/exl3/cache_precision.py'
        spec = importlib.util.spec_from_file_location('standalone_cache_precision', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.add_cache_precision_args(argparse.ArgumentParser()).parse_args(
            ['--cache-type', 'aster5', '--cache-policy', str(self.write())])
        self.assertEqual(module.resolve_cache_types(args), ('aster5', 'aster5'))
        with self.assertRaises(ValueError):
            module._cache_policy().validate_cache_policy(self.policy, b'wrong', self.descriptors())

    def test_excess_layers_and_unknown_components_rejected(self):
        for component in ({}, {'target': []}, {'target': [self.layer], 'typo': [self.layer]},
                          {'target': [dict(self.layer, layer_idx=i) for i in range(513)]}):
            with self.subTest(component_count=len(component)), self.assertRaises(ValueError):
                load_cache_policy(self.write(dict(self.policy, components=component)))

    def test_real_host_cli_preserves_quantized_backend_restrictions(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        for flag in ('--native-attention', '--draft-step-graph'):
            prerequisites = ['--gpu-embedding', '--gpu-draft', '--gpu-draft-metadata'] if flag == '--draft-step-graph' else []
            result = subprocess.run([sys.executable, str(root / 'run.py'), 'serve',
                                     '--cache-type', 'aster5', '--cache-policy', str(self.write()),
                                     '--mtp', '6', flag, *prerequisites], cwd=root, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)
            self.assertIn('Quantized KV cache cannot combine', result.stderr)


if __name__ == '__main__':
    unittest.main()
