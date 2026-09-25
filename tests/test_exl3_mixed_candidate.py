"""The pre-load completeness gate accepts dense donors without weakening packed checks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_exl3_model_smoke import check_required_tensors


class Linear(SimpleNamespace): pass
class Embedding(SimpleNamespace): pass
class RMSNorm(SimpleNamespace): pass
class GatedDeltaNet(SimpleNamespace): pass


class MixedCandidateTests(unittest.TestCase):
    def check(self, names, modules):
        config=SimpleNamespace(stc=SimpleNamespace(has_tensor=set(names).__contains__))
        classes=SimpleNamespace(Linear=Linear, Embedding=Embedding, RMSNorm=RMSNorm, GatedDeltaNet=GatedDeltaNet)
        with patch.dict(sys.modules, {'exllamav3.modules': classes}):
            check_required_tensors(config, [modules])

    def test_dense_mtp_with_conversion_group(self):
        self.check(['mtp.fc.weight'], [Linear(key='mtp.fc', qmap='mtp.fc')])

    def test_mixed_target_and_dense_donor(self):
        names=['target.'+s for s in ('trellis','suh','svh')]+['mtp.fc.weight']
        self.check(names, [Linear(key='target',qmap='target'), Linear(key='mtp.fc',qmap='mtp.fc')])

    def test_packed_donor_still_supported(self):
        self.check(['mtp.fc.'+s for s in ('trellis','suh','svh')], [Linear(key='mtp.fc',qmap='mtp.fc')])

    def test_each_missing_packed_member_fails(self):
        for missing in ('trellis','suh','svh'):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'mtp.fc.'+missing):
                self.check(['mtp.fc.'+s for s in ('trellis','suh','svh') if s!=missing], [Linear(key='mtp.fc',qmap='mtp.fc')])

    def test_partial_packed_does_not_hide_behind_dense_fallback(self):
        with self.assertRaisesRegex(ValueError, 'mtp.fc.suh'):
            self.check(['mtp.fc.weight','mtp.fc.trellis'], [Linear(key='mtp.fc',qmap='mtp.fc')])

    def test_missing_both_formats_fails(self):
        with self.assertRaisesRegex(ValueError, 'mtp.fc.trellis'):
            self.check([], [Linear(key='mtp.fc',qmap='mtp.fc')])

    def test_unquantized_projection_still_requires_weight(self):
        self.check(['gate.weight'], [Linear(key='gate',qmap=None)])
        with self.assertRaisesRegex(ValueError, 'gate.weight'):
            self.check([], [Linear(key='gate',qmap=None)])

    def test_embedding_and_norm_requirements_remain(self):
        modules=[Embedding(key='embed'), RMSNorm(unweighted=False,tensor_key='norm.weight')]
        self.check(['embed.weight','norm.weight'], modules)
        for missing in ('embed.weight','norm.weight'):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.check([name for name in ('embed.weight','norm.weight') if name!=missing], modules)


if __name__=='__main__':unittest.main()
