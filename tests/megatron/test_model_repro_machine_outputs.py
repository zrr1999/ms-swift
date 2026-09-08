# Copyright (c) ModelScope Contributors. All rights reserved.
"""Production machine-output and checkpoint-path contracts without GPU imports."""
import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PRINT = ROOT / 'swift/megatron/callbacks/print.py'
BASE = ROOT / 'swift/megatron/trainers/base.py'
namespace = {'os': os, 'Path': Path, 'hashlib': hashlib, 'json': json}
functions = [
    n for n in ast.parse(PRINT.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name in ('raw_loss_event', 'machine_loss_payload', '_write_json')
]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(PRINT), 'exec'), namespace)
trainer = next(
    n for n in ast.parse(BASE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'BaseMegatronTrainer')
save = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'save_checkpoint')
namespace.update(gc_collect=lambda: None, save_mcore_checkpoint=lambda *args, **kwargs: None, is_master=lambda: False)
exec(compile(ast.Module(body=[save], type_ignores=[]), str(BASE), 'exec'), namespace)


class MachineOutputTests(unittest.TestCase):

    def test_unrounded_training_losses_exclude_evaluation(self):
        event = namespace['raw_loss_event'](1, {'loss': 0.123456789123, 'mtp_1_loss': 0.5, 'eval_loss': 10})
        self.assertEqual(event, {'step': 1, 'loss': 0.123456789123, 'mtp_1_loss': 0.5})
        self.assertIsNone(namespace['raw_loss_event'](2, {'eval_loss': 10}))
        result = namespace['machine_loss_payload']([event])
        self.assertEqual(result['losses'], [0.123456789123])
        self.assertEqual(result['steps'], [1])
        self.assertNotIn('owning_cli_exit_code', result)

    def test_final_override_preserves_native_export_and_args_source(self):
        for iteration, override in [(99, True), (100, True), (100, False)]:
            with self.subTest(iteration=iteration, override=override), tempfile.TemporaryDirectory() as directory:
                output = str(Path(directory) / 'training')
                final = str(Path(directory) / 'canonical' / 'checkpoint')
                args = SimpleNamespace(
                    output_dir=output,
                    train_iters=100,
                    save_safetensors=True,
                    no_save_optim=True,
                    tuner_type='full',
                    merge_lora=False)
                state = SimpleNamespace(iteration=iteration, consumed_train_samples=iteration, best_global_step=None)
                copies, exports = [], []
                obj = SimpleNamespace(
                    args=args,
                    state=state,
                    optimizer=None,
                    opt_param_scheduler=None,
                    unwrapped_models=['native-model'],
                    template=SimpleNamespace(processor='processor'),
                    bridge=SimpleNamespace(save_weights=lambda *a, **kw: exports.append((a, kw))),
                    copy_path=lambda source, target: copies.append((source, target)))
                environment = {'MODEL_REPRO_CHECKPOINT_DIR': final} if override else {}
                with patch.dict(os.environ, environment, clear=True):
                    namespace['save_checkpoint'](obj)
                expected = final if override and iteration == 100 else str(Path(output) / f'checkpoint-{iteration}')
                self.assertEqual(state.last_model_checkpoint, expected)
                self.assertEqual(copies[0], (str(Path(output) / 'args.json'), str(Path(expected) / 'args.json')))
                self.assertEqual(exports[0][0], (['native-model'], expected))
                self.assertEqual(exports[0][1]['processor'], 'processor')


if __name__ == '__main__':
    unittest.main()
