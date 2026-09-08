"""Native criterion and DSA specification contracts without distributed setup."""

import ast
import contextlib
import io
import sys
import torch
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def production_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f'Expected one production function: {path}:{name}')
    node = matches[0]
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node],
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class AccuracyLossAndNormTest(unittest.TestCase):

    def test_mask_scale_local_gradient_and_global_reporting(self):
        torch.cuda.set_device(0)
        for enabled in (False, True):
            with self.subTest(accuracy=enabled):
                loss_func = production_function(
                    'swift/megatron/trainers/trainer.py', 'loss_func', {
                        'torch': torch,
                        'mpu': types.SimpleNamespace(get_data_parallel_group=lambda **kwargs: None),
                        '_use_accuracy_compatible_enabled': lambda: enabled,
                    })
                trainer = types.SimpleNamespace(
                    args=types.SimpleNamespace(enable_dft_loss=False, enable_channel_loss=False))
                values = torch.tensor([[2., 19., 3.]], device='cuda', requires_grad=True)
                labels = torch.tensor([[1, -100, 2]], device='cuda')
                scale = torch.tensor([[0.5, 1000., 2.]], device='cuda')
                # A second identical DP rank contributes to reporting, not local backward.
                with patch.object(torch.distributed, 'all_reduce', side_effect=lambda value, **kwargs: value.mul_(2)), \
                        patch.object(torch.distributed, 'get_rank', return_value=0), \
                        contextlib.redirect_stdout(io.StringIO()):
                    loss, count, metrics = loss_func(trainer, values, labels=labels, loss_scale=scale)
                self.assertEqual(loss.item(), 7.)
                self.assertEqual(count.item(), 2)
                self.assertEqual(metrics['loss'].tolist(), [14., 4.])
                self.assertFalse(metrics['loss'].requires_grad)
                loss.backward()
                self.assertEqual(values.grad.tolist(), [[0.5, 0., 2.]])

    def test_indexer_norm_preserves_disabled_provider(self):
        native_norm = type('NativeNorm', (), {})
        provider_norm = type('ProviderNorm', (), {})
        module = types.ModuleType('megatron.core.transformer.torch_norm')
        module.WrappedTorchNorm = native_norm
        for enabled, norm_accuracy in ((False, False), (True, False), (True, True)):
            with self.subTest(accuracy=enabled, norm_accuracy=norm_accuracy):
                calls = []
                replace_spec = production_function(
                    'swift/megatron/init.py', 'replace_spec_dsa', {
                        'origin_replace_spec_dsa': lambda *args: calls.append('provider'),
                        '_use_accuracy_compatible_enabled': lambda: enabled,
                    })
                indexer = types.SimpleNamespace(submodules=types.SimpleNamespace(k_norm=provider_norm))
                attention = types.SimpleNamespace(
                    submodules=types.SimpleNamespace(
                        q_layernorm=provider_norm,
                        kv_layernorm=provider_norm,
                        core_attention=types.SimpleNamespace(submodules=types.SimpleNamespace(indexer=indexer))))
                spec = types.SimpleNamespace(submodules=types.SimpleNamespace(self_attention=attention))
                loader = types.SimpleNamespace(config=types.SimpleNamespace(norm_accuracy_compatible=norm_accuracy))
                with patch.dict(sys.modules, {module.__name__: module}):
                    replace_spec(loader, spec)
                self.assertEqual(calls, ['provider'])
                self.assertIs(indexer.submodules.k_norm, native_norm if enabled else provider_norm)
                expected_qkv = native_norm if norm_accuracy else provider_norm
                self.assertIs(attention.submodules.q_layernorm, expected_qkv)
                self.assertIs(attention.submodules.kv_layernorm, expected_qkv)


if __name__ == '__main__':
    unittest.main()
