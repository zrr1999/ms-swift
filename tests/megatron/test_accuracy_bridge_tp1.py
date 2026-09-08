"""Unit tests for ms-swift TP1 accuracy bridge patch."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
_UAC = {"on": False}
_TP = {"size": 1}


def _use_accuracy_compatible_enabled():
    return _UAC["on"]


def _load_patch():
    src = (ROOT / "swift/megatron/init.py").read_text()
    tree = ast.parse(src)
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_mcore_bridge_tp1_accuracy"
    )
    target.decorator_list = []
    mod = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"_use_accuracy_compatible_enabled": _use_accuracy_compatible_enabled}
    exec(compile(mod, "swift/megatron/init.py", "exec"), ns)
    return ns["_patch_mcore_bridge_tp1_accuracy"]


_patch_mcore_bridge_tp1_accuracy = _load_patch()


def _make_original(name):
    def make_viewless_tensor(inp, requires_grad, keep_graph):
        make_viewless_tensor.calls.append((inp, requires_grad, keep_graph))
        return ("wrapped", inp)

    make_viewless_tensor.calls = []
    make_viewless_tensor.__name__ = name
    return make_viewless_tensor


class TestBridgeTp1Patch(unittest.TestCase):
    def setUp(self):
        self.orig_mtp = _make_original("mtp")
        self.orig_block = _make_original("block")
        self.mtp = ModuleType("mtp_layer")
        self.block = ModuleType("transformer_block")
        self.mtp.make_viewless_tensor = self.orig_mtp
        self.block.make_viewless_tensor = self.orig_block

        fake_ps = SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: _TP["size"]
        )
        fake_core = ModuleType("megatron.core")
        fake_core.parallel_state = fake_ps
        fake_mcore = ModuleType("mcore_bridge")
        fake_model = ModuleType("mcore_bridge.model")
        fake_modules = ModuleType("mcore_bridge.model.modules")
        fake_modules.mtp_layer = self.mtp
        fake_modules.transformer_block = self.block
        fake_megatron = ModuleType("megatron")

        p = patch.dict(
            sys.modules,
            {
                "megatron": fake_megatron,
                "megatron.core": fake_core,
                "mcore_bridge": fake_mcore,
                "mcore_bridge.model": fake_model,
                "mcore_bridge.model.modules": fake_modules,
            },
        )
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        _UAC["on"] = False
        _TP["size"] = 1

    def test_idempotent_and_both_module_globals(self):
        _patch_mcore_bridge_tp1_accuracy()
        first_mtp = self.mtp.make_viewless_tensor
        first_block = self.block.make_viewless_tensor
        self.assertTrue(getattr(first_mtp, "_swift_tp1_accuracy_patch", False))
        self.assertTrue(getattr(first_block, "_swift_tp1_accuracy_patch", False))
        _patch_mcore_bridge_tp1_accuracy()
        self.assertIs(self.mtp.make_viewless_tensor, first_mtp)
        self.assertIs(self.block.make_viewless_tensor, first_block)

    def test_uac_tp1_identity_both_modules(self):
        _patch_mcore_bridge_tp1_accuracy()
        _UAC["on"] = True
        _TP["size"] = 1
        inp = object()
        self.assertIs(self.mtp.make_viewless_tensor(inp, True, True), inp)
        self.assertIs(self.block.make_viewless_tensor(inp, False, False), inp)
        self.assertEqual(self.orig_mtp.calls, [])
        self.assertEqual(self.orig_block.calls, [])

    def test_off_and_actual_tp2_delegate_to_original(self):
        _patch_mcore_bridge_tp1_accuracy()
        inp = object()
        _UAC["on"] = False
        _TP["size"] = 1
        out = self.mtp.make_viewless_tensor(inp, True, True)
        self.assertEqual(out, ("wrapped", inp))
        self.assertEqual(self.orig_mtp.calls[-1], (inp, True, True))
        _UAC["on"] = True
        _TP["size"] = 2
        out2 = self.block.make_viewless_tensor(inp, False, True)
        self.assertEqual(out2, ("wrapped", inp))
        self.assertEqual(self.orig_block.calls[-1], (inp, False, True))


if __name__ == "__main__":
    unittest.main()
