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

    @classmethod
    def _forward_contract_specs(cls, boundary_set):
        if boundary_set == 'coarse':
            return None
        if boundary_set == 'layer0_fine':
            return dict(cls._LAYER0_FINE_FORWARD_MODULES)
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
        raw_path = os.path.join(rank_dir, f'{file_name}.bin')
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

    def _install_forward_contract_once(self):
        output_dir = os.environ.get('MODEL_REPRO_FORWARD_RECEIPT_DIR')
        if not output_dir or getattr(self, '_forward_contract_installed', False):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        boundary_set = os.environ.get('MODEL_REPRO_FORWARD_BOUNDARY_SET', 'coarse')
        fine_specs = self._forward_contract_specs(boundary_set)
        self._forward_contract_boundary_set = boundary_set
        handles = []
        if fine_specs is not None:
            selected = []
            if rank < 2:
                module_hits = {name: [] for name in fine_specs}
                for chunk_index, model in enumerate(self.unwrapped_models):
                    for module_name, module in model.named_modules():
                        if module_name in module_hits:
                            module_hits[module_name].append((chunk_index, module))
                invalid = {name: len(hits) for name, hits in module_hits.items() if len(hits) != 1}
                if invalid:
                    raise RuntimeError(f'layer0 fine forward selectors must match exactly once on rank {rank}: {invalid}')
                for module_name, boundary in fine_specs.items():
                    chunk_index, module = module_hits[module_name][0]
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
                    boundary = 'output_head_output'
                elif module_name.startswith('mtp.layers.0.') and module_name.rsplit('.', 1)[-1] in {
                        'enorm', 'hnorm', 'eh_proj', 'mtp_model_layer', 'layer_norm', 'final_layernorm'}:
                    boundary = f"mtp_{module_name.removeprefix('mtp.layers.0.').replace('.', '_')}_output"
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
        parameters = []
        for chunk_index, model in enumerate(self.unwrapped_models):
            for name, param in model.named_parameters():
                parameters.append({
                    'chunk': chunk_index,
                    'name': name,
                    **self._parameter_record(param),
                })
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
