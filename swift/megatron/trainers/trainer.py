# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import os
import torch
import torch.distributed as dist
import torch.nn
from collections import defaultdict
from functools import partial
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional

from swift.utils import get_logger
from ..init import _use_accuracy_compatible_enabled
from .base import BaseMegatronTrainer

logger = get_logger()


def project_owning_loader_semantics(input_values, model_label_values, semantic_length, labels_were_shifted=True):
    """Normalize padded Megatron carrier tensors back to the dataset semantic row."""
    semantic_length = int(semantic_length)
    if semantic_length <= 0 or semantic_length > len(input_values) or semantic_length > len(model_label_values):
        raise ValueError(
            f'invalid owning-loader semantic length {semantic_length} for carrier lengths '
            f'{len(input_values)}/{len(model_label_values)}')
    semantic_input_values = input_values[:semantic_length]
    normalized_label_values = model_label_values
    if labels_were_shifted and model_label_values:
        # get_batch_on_this_pp_rank rolls causal-LM labels left by one before
        # model forward. Reverse that roll for a framework-neutral dataset receipt.
        normalized_label_values = model_label_values[-1:] + model_label_values[:-1]
    semantic_label_values = normalized_label_values[:semantic_length]
    semantic_mask_values = [label != -100 for label in semantic_label_values]
    return semantic_input_values, semantic_label_values, semantic_mask_values


# E-237: canonical names for the MTP transformer layer's internal boundaries.
# See the identical table on the other side; the map is what makes this a by-name
# pairing (E-216) despite the two frameworks' different submodule names.
_MTP_LAYER_INTERNAL_BOUNDARIES = {
    'mtp.layers.0.mtp_model_layer.input_layernorm': 'mtplayer_input_layernorm_output',
    'mtp.layers.0.mtp_model_layer.self_attention': 'mtplayer_self_attention_output',
    'mtp.layers.0.mtp_model_layer.self_attention.linear_proj': 'mtplayer_out_proj_output',
    'mtp.layers.0.mtp_model_layer.pre_mlp_layernorm': 'mtplayer_post_attn_norm_output',
    'mtp.layers.0.mtp_model_layer.mlp': 'mtplayer_mlp_output',
    'mtp.layers.0.mtp_model_layer.mlp.router': 'mtplayer_gate_output',
    'mtp.layers.0.mtp_model_layer.mlp.shared_experts': 'mtplayer_shared_experts_output',
    # E-238: split the MoE backward into its two branches; see the other side.
    'mtp.layers.0.mtp_model_layer.mlp.experts': 'mtplayer_routed_experts_output',
    # E-239: inside the shared expert; see the other side.
    'mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1': 'mtpshared_fc1_output',
    'mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2': 'mtpshared_fc2_output',
}

_MTP_LAYER_INTERNAL_BRANCH_INPUTS = {
    'mtp.layers.0.mtp_model_layer.mlp.experts': 'mtplayer_routed_experts_input',
    'mtp.layers.0.mtp_model_layer.mlp.shared_experts': 'mtplayer_shared_experts_input',
    'mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2': 'mtpshared_fc2_input',
    # E-259: ThreePath third clone is the router-path hidden. The existing
    # mtplayer_gate_output hook records router OUTPUT (topk scores), not d(hidden).
    'mtp.layers.0.mtp_model_layer.mlp.router': 'mtplayer_router_hidden_input',
}


class MegatronTrainer(BaseMegatronTrainer):

    _LAYER0_FINE_FORWARD_MODULES = {
        'decoder.layers.0.input_layernorm': 'layer0_input_rmsnorm_output',
        'decoder.layers.0.self_attention.linear_q_down_proj': 'layer0_q_down_projection_output',
        'decoder.layers.0.self_attention.q_layernorm': 'layer0_q_rmsnorm_output',
        'decoder.layers.0.self_attention.linear_q_up_proj': 'layer0_q_up_projection_output',
        'decoder.layers.0.self_attention.linear_kv_down_proj': 'layer0_kv_down_projection_output',
        'decoder.layers.0.self_attention.kv_layernorm': 'layer0_kv_rmsnorm_output',
        'decoder.layers.0.self_attention.linear_kv_up_proj': 'layer0_kv_up_projection_output',
        'decoder.layers.0.self_attention.linear_proj': 'layer0_attention_output_projection',
        'decoder.layers.0.self_attention': 'layer0_self_attention_output',
        'decoder.layers.0.mlp.linear_fc1': 'layer0_dense_fc1_output',
        'decoder.layers.0.mlp.linear_fc2': 'layer0_dense_fc2_output',
        'decoder.layers.0.mlp': 'layer0_dense_mlp_output',
        'decoder.layers.0': 'base_transformer_layer_0_output',
    }

    # E-051R: dynamic all-layers fine spec (4 layers, PP-aware) built per rank.
    _ALL_LAYERS_FINE_MODULE_SUFFIXES = {
        'input_layernorm': 'input_rmsnorm_output',
        'self_attention.linear_q_down_proj': 'q_down_projection_output',
        'self_attention.q_layernorm': 'q_rmsnorm_output',
        'self_attention.linear_q_up_proj': 'q_up_projection_output',
        'self_attention.linear_kv_down_proj': 'kv_down_projection_output',
        'self_attention.kv_layernorm': 'kv_rmsnorm_output',
        'self_attention.linear_kv_up_proj': 'kv_up_projection_output',
        'self_attention.linear_proj': 'attention_output_projection',
        'self_attention': 'self_attention_output',
        'mlp.linear_fc1': 'dense_fc1_output',
        'mlp.linear_fc2': 'dense_fc2_output',
        'mlp': 'dense_mlp_output',
        '': 'base_transformer_layer_output',
    }

    @classmethod
    def _all_layers_fine_specs(cls):
        """Per-rank 4-layer fine specs using PP2 mapping.

        Each PP stage holds local layers 0..1; stage 0 (rank<2) maps them to
        global 0..1, stage 1 (rank>=2) maps them to global 2..3.
        """
        import torch
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        offset = 0 if rank < 2 else 2
        specs = {}
        for local in (0, 1):
            global_layer = local + offset
            for suffix, boundary_suffix in cls._ALL_LAYERS_FINE_MODULE_SUFFIXES.items():
                module_name = f'decoder.layers.{local}'
                if suffix:
                    module_name += '.' + suffix
                boundary = f'layer{global_layer}_{boundary_suffix}'
                specs[module_name] = boundary
        return specs

    @classmethod
    def _layer3_moe_fine_specs(cls):
        """E-095: MoE-internal boundaries for global layer 3.

        On PP stage 1 the global layer 3 is the LOCAL layer 1, so the module
        prefix is ``decoder.layers.1``. Selectors that do not match exactly once
        are skipped by the installer, and the emitted metadata ``selectors`` list
        documents which module names actually exist.
        """
        prefix = 'decoder.layers.1'
        specs = {
            f'{prefix}.pre_mlp_layernorm': 'layer3_pre_mlp_rmsnorm_output',
            f'{prefix}.mlp': 'layer3_moe_mlp_output',
            f'{prefix}.mlp.router': 'layer3_router_output',
            f'{prefix}.mlp.shared_experts': 'layer3_shared_expert_output',
            f'{prefix}.mlp.shared_experts.linear_fc1': 'layer3_shared_expert_fc1_output',
            f'{prefix}.mlp.shared_experts.linear_fc2': 'layer3_shared_expert_fc2_output',
            f'{prefix}.mlp.experts': 'layer3_experts_output',
        }
        for expert in range(16):
            specs[f'{prefix}.mlp.experts.local_experts.{expert}'] = f'layer3_expert{expert}_output'
            specs[f'{prefix}.mlp.experts.local_experts.{expert}.linear_fc1'] = (
                f'layer3_expert{expert}_fc1_output')
            specs[f'{prefix}.mlp.experts.local_experts.{expert}.linear_fc2'] = (
                f'layer3_expert{expert}_fc2_output')
        return specs

    @classmethod
    def _forward_contract_specs(cls, boundary_set):
        if boundary_set == 'coarse':
            return None
        if boundary_set == 'layer0_fine':
            return dict(cls._LAYER0_FINE_FORWARD_MODULES)
        if boundary_set == 'all_layers_fine':
            return cls._all_layers_fine_specs()
        if boundary_set == 'layer3_moe_fine':
            return cls._layer3_moe_fine_specs()
        raise ValueError(f'unsupported MODEL_REPRO_FORWARD_BOUNDARY_SET: {boundary_set}')

    @staticmethod
    def _parameter_record(param):
        if hasattr(param, 'is_dist') and param.is_dist():
            param = param._local_value()
        tensor = param.detach().contiguous().to(device='cpu')

        def raw_digest(value):
            return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        zero_count = 0
        negative_zero_count = 0
        if tensor.is_floating_point():
            zero_mask = tensor == 0
            zero_count = int(zero_mask.sum().item())
            negative_zero_count = int((zero_mask & torch.signbit(tensor)).sum().item())
        record = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': tensor.numel(),
            'sha256': raw_digest(tensor),
            'positive_zero_count': zero_count - negative_zero_count,
            'negative_zero_count': negative_zero_count,
        }
        if tensor.ndim == 2:
            record['transpose_sha256'] = raw_digest(tensor.transpose(0, 1))
        return record

    @staticmethod
    def _first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = MegatronTrainer._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = MegatronTrainer._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _write_forward_kwarg_records(self, prefix, kwargs, names):
        """Record selected keyword arguments of a module call.

        A positional forward pre-hook only sees ``args``, so a layer that receives its
        rope and mask by keyword has those inputs invisible to it. E-219 needs exactly
        those: it eliminated the attention-mask axis for the MTP layer and promoted the
        rotary embeddings, which are passed by keyword.

        Each present, non-None entry is written under ``<prefix>_<name>`` so it pairs by
        NAME with the counterpart side, never by call order.
        """
        for name in names:
            value = kwargs.get(name)
            if value is None:
                continue
            self._write_forward_record(f'{prefix}_{name}', value)

    def _write_forward_record(self, boundary, value):
        tensor = self._first_tensor(value)
        if tensor is None:
            return
        output_dir = os.environ.get('MODEL_REPRO_FORWARD_RECEIPT_DIR')
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank_dir = os.path.join(output_dir, f'rank{rank}')
        os.makedirs(rank_dir, exist_ok=True)
        records = getattr(self, '_forward_contract_records', {})
        call_index = sum(name == boundary or name.startswith(f'{boundary}_call') for name in records)
        name = boundary if call_index == 0 else f'{boundary}_call{call_index}'
        tensor = tensor.detach().contiguous().to(device='cpu')
        raw = tensor.view(torch.uint8).numpy().tobytes()
        file_name = ''.join(character if character.isalnum() or character in '-_' else '_' for character in name)
        raw_path = os.path.join(rank_dir, f'{file_name}_s{os.environ.get("MODEL_REPRO_STEP", "x")}.bin')
        with open(raw_path, 'wb') as stream:
            stream.write(raw)
        zero_count = 0
        negative_zero_count = 0
        if tensor.is_floating_point():
            zero_mask = tensor == 0
            zero_count = int(zero_mask.sum().item())
            negative_zero_count = int((zero_mask & torch.signbit(tensor)).sum().item())
        records[name] = {
            'boundary': boundary,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': tensor.numel(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'positive_zero_count': zero_count - negative_zero_count,
            'negative_zero_count': negative_zero_count,
            'raw_path': raw_path,
        }
        self._forward_contract_records = records
        self._install_backward_record(name, self._first_tensor(value))
        payload = {
            'schema': 'glm52-local-forward-boundaries/v1',
            'framework': 'torch',
            'rank': rank,
            'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            'boundary_set': getattr(self, '_forward_contract_boundary_set', 'coarse'),
            'selectors': getattr(self, '_forward_contract_selector_receipt', []),
            'records': records,
        }
        with open(os.path.join(rank_dir, 'metadata.json'), 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')

    def _install_backward_record(self, name, tensor):
        """Capture d(loss)/d(boundary output) for an already-recorded boundary.

        Forward boundaries are bit-exact through the dense stack while dense
        weight gradients still disagree well above bf16 rounding, so the
        divergence enters during backward. Hooking the very tensor the forward
        receipt stored keeps both receipts on one boundary definition.
        """
        output_dir = os.environ.get('MODEL_REPRO_BACKWARD_RECEIPT_DIR')
        if not output_dir or tensor is None or not tensor.requires_grad:
            return
        tensor.register_hook(lambda grad, name=name: self._write_backward_record(name, grad))

    def _write_backward_record(self, name, grad):
        output_dir = os.environ.get('MODEL_REPRO_BACKWARD_RECEIPT_DIR')
        if not output_dir or grad is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank_dir = os.path.join(output_dir, f'rank{rank}')
        os.makedirs(rank_dir, exist_ok=True)
        records = getattr(self, '_backward_contract_records', {})
        # A boundary tensor can receive several gradient contributions; number
        # them so a differing accumulation order stays visible instead of being
        # overwritten by the last writer.
        call_index = sum(key == name or key.startswith(f'{name}_bwd') for key in records)
        key = name if call_index == 0 else f'{name}_bwd{call_index}'
        grad = grad.detach().contiguous().to(device='cpu')
        raw = grad.view(torch.uint8).numpy().tobytes()
        file_name = ''.join(character if character.isalnum() or character in '-_' else '_' for character in key)
        raw_path = os.path.join(rank_dir, f'{file_name}_s{os.environ.get("MODEL_REPRO_STEP", "x")}.bin')
        with open(raw_path, 'wb') as stream:
            stream.write(raw)
        records[key] = {
            'boundary': name,
            'shape': list(grad.shape),
            'dtype': str(grad.dtype),
            'numel': grad.numel(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'raw_path': raw_path,
        }
        self._backward_contract_records = records
        payload = {
            'schema': 'glm52-local-backward-boundaries/v1',
            'framework': 'torch',
            'rank': rank,
            'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            'boundary_set': getattr(self, '_forward_contract_boundary_set', 'coarse'),
            'records': records,
        }
        with open(os.path.join(rank_dir, 'metadata.json'), 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')

    def _dump_constructed_config(self, rank):
        """E-252: serialise the config actually used to build the layers.

        The bias_activation_fusion asymmetry that caused the dense shared-expert backward
        residual was provider-hardcoded on the other side, absent from both task YAMLs,
        and found only because an instrumentation dump stayed silent. A field-by-field
        diff of the two frameworks' CONSTRUCTED configs finds that class of problem
        directly. Off unless MODEL_REPRO_CONFIG_DUMP is set.
        """
        output_dir = os.environ.get('MODEL_REPRO_CONFIG_DUMP')
        if not output_dir:
            return
        config = None
        for model in getattr(self, 'unwrapped_models', []) or []:
            config = getattr(model, 'config', None)
            if config is not None:
                break
        if config is None:
            logger.warning('[CONFIG-DUMP] no model config found')
            return

        def normalise(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, (list, tuple)):
                return [normalise(v) for v in value]
            if isinstance(value, dict):
                return {str(k): normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            name = getattr(value, '__name__', None)
            if name:
                keywords = getattr(value, 'keywords', None)
                return f'{name}({normalise(keywords)})' if keywords else name
            if hasattr(value, 'func'):
                return f'partial({normalise(value.func)},{normalise(getattr(value, "keywords", None))})'
            return f'<{type(value).__name__}>'

        fields = {}
        for key in dir(config):
            if key.startswith('_'):
                continue
            try:
                value = getattr(config, key)
            except Exception:
                continue
            if getattr(value, '__self__', None) is not None:
                continue
            if callable(value) and key not in {
                    'activation_func', 'hidden_act', 'init_method', 'output_layer_init_method'}:
                continue
            fields[key] = normalise(value)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'torch_rank{rank}.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump({'schema': 'glm52-constructed-config/v1', 'framework': 'torch',
                       'rank': rank, 'fields': fields}, stream, ensure_ascii=False,
                      indent=2, sort_keys=True)
            stream.write('\n')
        logger.info(f'[CONFIG-DUMP] wrote {path} with {len(fields)} fields')

    def _dump_module_types_once(self, rank):
        """E-259: record live module classes; fail closed if TE is still in the graph."""
        if getattr(self, '_module_type_dumped', False):
            return
        rows = []
        te_names = []
        for chunk_index, model in enumerate(self.unwrapped_models):
            for name, module in model.named_modules():
                cls = type(module)
                mod = cls.__module__
                is_te = 'transformer_engine' in mod
                if is_te:
                    te_names.append(name or cls.__name__)
                    rows.append({
                        'chunk': chunk_index,
                        'name': name,
                        'class': f'{mod}.{cls.__name__}',
                        'is_te': True,
                    })
                    continue
                if 'torch.nn' not in mod and 'megatron' not in mod:
                    continue
                interesting = (
                    'linear' in name.lower()
                    or 'norm' in name.lower()
                    or name.endswith('self_attention')
                    or name.endswith('core_attention')
                    or 'indexer' in name
                    or name.endswith('pre_mlp_layernorm')
                    or name.endswith('input_layernorm')
                    or name.endswith('router')
                )
                if not interesting:
                    continue
                rows.append({
                    'chunk': chunk_index,
                    'name': name,
                    'class': f'{mod}.{cls.__name__}',
                    'is_te': False,
                })
        self._module_type_dumped = True
        te_count = len(te_names)
        output_dir = os.environ.get('MODEL_REPRO_MODULE_TYPE_DUMP')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f'torch_rank{rank}.json')
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump({
                    'schema': 'glm52-module-types/v1',
                    'framework': 'torch',
                    'rank': rank,
                    'te_module_count': te_count,
                    'modules': rows,
                }, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write('\n')
            logger.info(f'[MODULE-TYPE-DUMP] wrote {path} te_count={te_count}')
        if _use_accuracy_compatible_enabled() and te_count:
            raise RuntimeError(
                'use_accuracy_compatible requires Transformer Engine off on this '
                'side (PaddleFleet HAVE_TE is False; E-259: remaining DSA TE '
                f'moved post_attn_norm). Found {te_count} TE modules: {te_names[:12]}'
            )

    def _install_forward_contract_once(self):
        output_dir = os.environ.get('MODEL_REPRO_FORWARD_RECEIPT_DIR')
        if not output_dir or getattr(self, '_forward_contract_installed', False):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._dump_constructed_config(rank)
        self._dump_module_types_once(rank)
        boundary_set = os.environ.get('MODEL_REPRO_FORWARD_BOUNDARY_SET', 'coarse')
        fine_specs = self._forward_contract_specs(boundary_set)
        self._forward_contract_boundary_set = boundary_set
        handles = []
        # E-051R Explore: env-gated live absorbed / pre-o_proj / attn_out / moe dump (default OFF).
        if (os.environ.get('MODEL_REPRO_TORCH_DUMP_ABSORBED', '0') == '1'
                or os.environ.get('MODEL_REPRO_TORCH_DUMP_PRE_OPROJ', '0') == '1'
                or os.environ.get('MODEL_REPRO_TORCH_DUMP_ATTN_OUT', '0') == '1'
                or os.environ.get('MODEL_REPRO_TORCH_DUMP_MOE_OUT', '0') == '1'):
            try:
                from torch_live_absorbed_capture import install_on_models

                n_abs = install_on_models(self.unwrapped_models)
                logger.info(f'[torch-live-absorbed/pre-oproj/attn-out/moe] installed hooks={n_abs} rank={rank}')
            except Exception as e:
                logger.warning(f'[torch-live-absorbed/pre-oproj/attn-out/moe] install failed: {e}')
        if fine_specs is not None:
            selected = []
            module_hits = {name: [] for name in fine_specs}
            for chunk_index, model in enumerate(self.unwrapped_models):
                for module_name, module in model.named_modules():
                    if module_name in module_hits:
                        module_hits[module_name].append((chunk_index, module))
            invalid = {name: len(hits) for name, hits in module_hits.items() if len(hits) != 1}
            if invalid:
                # Hybrid MoE models: some layers are MoE (no linear_fc1/fc2) or
                # otherwise structurally different; skip unmatched selectors.
                logger.warning(f'fine forward selectors unmatched on rank {rank}: {invalid}')
            module_hits_ok = {name: hits for name, hits in module_hits.items() if len(hits) == 1}
            for module_name, boundary in fine_specs.items():
                hits = module_hits_ok.get(module_name)
                if not hits:
                    continue
                chunk_index, module = hits[0]
                handles.append(module.register_forward_hook(
                    lambda _module, _inputs, output, name=boundary: self._write_forward_record(name, output)))
                selected.append({'chunk': chunk_index, 'module': module_name, 'boundary': boundary})
            self._forward_contract_selector_receipt = selected
            rank_dir = os.path.join(output_dir, f'rank{rank}')
            os.makedirs(rank_dir, exist_ok=True)
            with open(os.path.join(rank_dir, 'metadata.json'), 'w', encoding='utf-8') as stream:
                json.dump({
                    'schema': 'glm52-local-forward-boundaries/v1',
                    'framework': 'torch',
                    'rank': rank,
                    'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                    'boundary_set': boundary_set,
                    'selectors': selected,
                    'records': {},
                }, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write('\n')
            self._forward_contract_handles = handles
            self._forward_contract_installed = True
            return
        for chunk_index, model in enumerate(self.unwrapped_models):
            for module_name, module in model.named_modules():
                boundary = None
                match = __import__('re').fullmatch(r'decoder\.layers\.(\d+)', module_name)
                if module_name == 'embedding':
                    boundary = f'chunk{chunk_index}_embedding_output'
                elif match:
                    local_layer = int(match.group(1))
                    global_layer = local_layer if rank < 2 else local_layer + 2
                    input_boundary = f'base_layer_{global_layer}_input'
                    handles.append(module.register_forward_pre_hook(
                        lambda _module, inputs, name=input_boundary: self._write_forward_record(name, inputs)))
                    boundary = f'base_layer_{global_layer}_output'
                elif module_name == 'decoder.final_layernorm':
                    boundary = 'final_norm_output'
                elif module_name == 'output_layer':
                    input_boundary = 'output_head_input'
                    handles.append(module.register_forward_pre_hook(
                        lambda _module, inputs, name=input_boundary: self._write_forward_record(name, inputs)))
                    # E-229/E-230: no new hook is needed to see the gradient arriving at
                    # the head. _write_forward_record already calls
                    # _install_backward_record on the very tensor it stored, so under
                    # MODEL_REPRO_BACKWARD_RECEIPT_DIR the boundary 'output_head_output'
                    # yields dY -- the gradient of the logits, straight out of the
                    # vocab-parallel cross-entropy backward -- and 'output_head_input'
                    # yields the head dgrad.
                    #
                    # Those two are what E-230 needs. E-228 proved the residual is
                    # backward-only by construction (identical official weights, bit-equal
                    # step-1 forward on both loss scalars, divergent step 2), and E-229
                    # found it is systematically one-sided (59 of 64 comparable families
                    # have a smaller paddle gradient, sign-test P = 4.5e-13) and already
                    # present at output_layer.weight, the first weight gradient produced
                    # anywhere in the true global backward order. Since dW = dY^T X with X
                    # bit-exact forward, dY separates "the loss backward differs" from
                    # "the weight-gradient accumulation differs".
                    #
                    # A module-level register_full_backward_hook was tried first and
                    # discarded: this module returns a (logits, bias) tuple, which that API
                    # does not support, and it would have duplicated a channel that is
                    # already symmetric with the paddle side.
                    boundary = 'output_head_output'
                elif module_name.startswith('mtp.layers.0.') and module_name.rsplit('.', 1)[-1] in {
                        'enorm', 'hnorm', 'eh_proj', 'mtp_model_layer', 'layer_norm', 'final_layernorm'}:
                    leaf = module_name.rsplit('.', 1)[-1]
                    boundary = f"mtp_{module_name.removeprefix('mtp.layers.0.').replace('.', '_')}_output"
                    # enorm/hnorm inputs are the two MTP branch entry tensors
                    # (the rolled embedding and the trunk hidden state). They are not
                    # module outputs anywhere, so without a pre-hook the branch has no
                    # observable input and a difference at enorm_output cannot be
                    # attributed to the norm rather than to what was fed to it.
                    # mtp_model_layer's input closes the last gap: E-218 made the
                    # whole branch bit-exact up to and including eh_proj while the
                    # layer's OUTPUT still differed, so the layer's own input is the
                    # only remaining place the difference can enter.
                    if leaf in {'enorm', 'hnorm', 'mtp_model_layer'}:
                        input_boundary = f'mtp_{leaf}_input'
                        handles.append(module.register_forward_pre_hook(
                            lambda _module, inputs, name=input_boundary: self._write_forward_record(name, inputs)))
                    if leaf == 'mtp_model_layer':
                        # E-219 eliminated the attention-mask axis, leaving the rotary
                        # embeddings as the leading candidate: this side reuses the trunk
                        # rope while PaddleFleet re-trims it for MTP. The layer is called
                        # with keyword arguments, so with_kwargs is required to see them;
                        # the positional pre-hook above only ever sees hidden_states.
                        handles.append(module.register_forward_pre_hook(
                            lambda _module, _args, kwargs: self._write_forward_kwarg_records(
                                'mtp_model_layer', kwargs,
                                ('rotary_pos_emb', 'rotary_pos_cos', 'rotary_pos_sin',
                                 'attention_mask', 'attention_bias')),
                            with_kwargs=True))
                elif module_name in _MTP_LAYER_INTERNAL_BRANCH_INPUTS:
                    # E-238: both MoE branches are fed the same tensor, so their
                    # input gradients are what separates them.
                    input_boundary = _MTP_LAYER_INTERNAL_BRANCH_INPUTS[module_name]
                    handles.append(module.register_forward_pre_hook(
                        lambda _module, inputs, name=input_boundary: self._write_forward_record(name, inputs)))
                    boundary = _MTP_LAYER_INTERNAL_BOUNDARIES.get(module_name)
                elif module_name in _MTP_LAYER_INTERNAL_BOUNDARIES:
                    # E-237: the MTP transformer layer is ENTERED with a bit-equal
                    # gradient and LEFT with a differing one (E-236), while its
                    # forward is bit-exact. These are its internal boundaries,
                    # named canonically so the two frameworks' differing module
                    # names pair up by NAME, not by emission order (E-216).
                    boundary = _MTP_LAYER_INTERNAL_BOUNDARIES[module_name]
                if boundary is not None:
                    handles.append(module.register_forward_hook(
                        lambda _module, _inputs, output, name=boundary: self._write_forward_record(name, output)))
        self._forward_contract_handles = handles
        self._forward_contract_installed = True

    def _write_parameter_contract_once(self):
        output_dir = os.environ.get('MODEL_REPRO_PARAMETER_RECEIPT_DIR')
        if not output_dir or getattr(self, '_parameter_contract_written', False):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        parameter_match = os.environ.get('MODEL_REPRO_PARAMETER_RECEIPT_MATCH')
        parameters = []
        for chunk_index, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                if parameter_match and parameter_match not in name:
                    continue
                parameters.append({
                    'chunk': chunk_index,
                    'name': name,
                    **self._parameter_record(param),
                })
        raw_dir = os.environ.get('MODEL_REPRO_PARAMETER_RAW_DIR')
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
            for item in parameters:
                safe = item['name'].replace('/', '_').replace('.', '_')
                for chunk_index, model in enumerate(self.unwrapped_models):
                    for name, param in model.named_parameters():
                        if name == item['name']:
                            param.detach().float().cpu().numpy().tofile(os.path.join(raw_dir, f'chunk{chunk_index}_rank{rank}_{safe}.f32.bin'))
        payload = {
            'schema': 'glm52-loaded-parameter-inventory/v1',
            'framework': 'torch',
            'rank': rank,
            'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            'parameters': parameters,
            'parameter_count': len(parameters),
            'local_numel': sum(item['numel'] for item in parameters),
        }
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'rank{rank}.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        self._parameter_contract_written = True

    def _write_input_contract_once(self, data, seq_lens=None):
        self._write_parameter_contract_once()
        self._install_forward_contract_once()
        if _use_accuracy_compatible_enabled() and not getattr(self, '_module_type_dumped', False):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._dump_module_types_once(rank)
        path = os.environ.get('MODEL_REPRO_INPUT_RECEIPT_PATH')
        if not path or getattr(self, '_input_contract_written', False):
            return
        if not mpu.is_pipeline_last_stage(ignore_virtual=False):
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != torch.distributed.get_world_size() - 1:
            return
        input_ids = data.get('input_ids')
        labels = data.get('labels')
        if input_ids is None or labels is None:
            return

        def values(tensor):
            return tensor.detach().to(device='cpu', dtype=torch.int64).reshape(-1).tolist()

        def digest(items):
            return hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()

        input_values = values(input_ids)
        label_values = values(labels)
        model_mask_values = [label != -100 for label in label_values]
        semantic_length = seq_lens[0] if seq_lens else len(input_values)
        labels_were_shifted = self.args.task_type == 'causal_lm' and not getattr(
            self.args, 'pretokenized_dataset', False)
        semantic_input_values, semantic_label_values, semantic_mask_values = project_owning_loader_semantics(
            input_values, label_values, semantic_length, labels_were_shifted)
        payload = {
            'schema': 'glm52-owning-loader-input/v1',
            'framework': 'torch',
            'rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            'step': self.state.iteration + 1,
            'input_ids': {
                'shape': list(input_ids.shape),
                'dtype': str(input_ids.dtype),
                'count': len(input_values),
                'sha256': digest(input_values),
            },
            'labels': {
                'shape': list(labels.shape),
                'dtype': str(labels.dtype),
                'count': len(label_values),
                'supervised_count': sum(model_mask_values),
                'sha256': digest(label_values),
                'projection': 'model_next_token_labels',
            },
            'loss_mask': {
                'shape': list(labels.shape),
                'dtype': 'bool',
                'count': len(model_mask_values),
                'supervised_count': sum(model_mask_values),
                'sha256': digest(model_mask_values),
            },
            'semantic': {
                'input_token_count': len(semantic_input_values),
                'supervised_target_count': sum(semantic_mask_values),
                'input_ids_sha256': digest(semantic_input_values),
                'labels_sha256': digest(semantic_label_values),
                'loss_mask_sha256': digest(semantic_mask_values),
                'projection': 'dataset_row_before_megatron_padding_and_label_roll',
            },
            'carrier_padding': {
                'count': len(input_values) - len(semantic_input_values),
                'input_ids_sha256': digest(input_values[len(semantic_input_values):]),
                'labels_sha256': digest(label_values[len(semantic_input_values):]),
            },
            'ignore_index': -100,
            'dataset': os.environ.get('MODEL_REPRO_INPUT_DATASET_PATH'),
        }
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        self._input_contract_written = True

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
        logits = self.get_last_tokens(output_tensor, packed_seq_params, attention_mask)
        num_labels = args.num_labels
        acc = None
        if args.problem_type == 'regression':
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif args.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            acc = (logits.detach().argmax(dim=-1) == labels).float().mean()
        elif args.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            preds = logits.sigmoid() > 0.5
            acc = (labels == preds).all(dim=-1).float().mean()
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None):
        args = self.args

        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        # Reduce loss for logging.
        reporting_loss = loss.detach().clone()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group(with_context_parallel=True))

        lm_loss = loss[0]
        lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)


        if _use_accuracy_compatible_enabled():
            # 精度对齐锚点 2（对应 PF language_loss.py forward_impl 出口的 final_loss）：
            # 本 rank、本 micro-batch 的 sum(loss*mask)/valid_tokens，
            # 未跨 DP all-reduce、未除 num_microbatches，与 PF 侧同语义。
            import hashlib as _hashlib
            _final = (loss[0].detach().float() / loss[1].detach().float().clamp(min=1)).contiguous()
            print(
                f"\nfinal_loss: rank={torch.distributed.get_rank()} "
                f"val={_final.item():.20f} "
                f"md5={_hashlib.md5(_final.cpu().numpy().tobytes()).hexdigest()}",
                flush=True)

        metrics = {'loss': reporting_loss}
        if args.enable_channel_loss:
            metrics.update(self._compute_channel_loss(losses, loss_mask, channels, packed_seq_params))
        return (lm_loss, local_num_tokens, metrics)

    def _compute_channel_loss(self, losses, loss_mask, channels, packed_seq_params=None):
        args = self.args
        metrics = defaultdict(lambda: torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device()))
        if args.padding_free:
            num_samples = packed_seq_params.seq_lens.shape[0]
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            for i in range(cu_seqlens.shape[0] - 1):
                channel = None if channels is None else channels[i]
                slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                c_loss = losses[0, slice_][loss_mask[0, slice_]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]
        else:
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                c_loss = losses[i][loss_mask[i]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]

        # Synchronize keys to avoid getting stuck.
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_keys = [None] * torch.distributed.get_world_size(group=dp_cp_group)
        dist.all_gather_object(all_keys, list(metrics.keys()), group=dp_cp_group)
        new_metrics = {}
        for key in sorted(set().union(*all_keys)):
            new_metrics[key] = metrics[key]
        new_metrics = self._all_reduce_metric(new_metrics, torch.distributed.ReduceOp.SUM, group=dp_cp_group)
        return new_metrics

    def forward_step(self, data_iterator, model):
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        seq_lens = data.pop('_model_repro_seq_lens', None)
        self._write_input_contract_once(data, seq_lens)
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        if self.args.task_type == 'seq_cls':
            loss_func = partial(
                self.seq_cls_loss_func,
                labels=labels,
                packed_seq_params=packed_seq_params,
                attention_mask=data.get('attention_mask')
                if data.get('attention_mask') is not None else data.get('attention_mask_2d'))
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params)
        return output_tensor, loss_func
