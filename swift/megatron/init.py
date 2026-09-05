# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import importlib.metadata
import logging
import os
import sys
import torch
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy, deepcopy
from packaging import version
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version
from typing import Optional

from swift.model import get_model_processor, save_checkpoint
from swift.utils import (HfConfigFactory, disable_safe_ddp_context_use_barrier, get_logger, get_modules_to_not_convert,
                         get_multimodal_target_regex, is_master, split_list)

logger = get_logger()


def _get_save_processor_id(args):
    """Use the configured processor source when weights and tokenizer are independent."""
    return getattr(args, 'tokenizer_name_or_path', None) or args.model_dir


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future

    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = (validate_non_overlapping_shards_metadata)
    api2.validate_non_overlapping_shards_metadata = (validate_non_overlapping_shards_metadata)

    def _validate_global_plan(*args, **kwargs):
        # torch returns a list of error messages here; empty list means "no error".
        return []

    default_planner._validate_global_plan = _validate_global_plan


def _patch_transformers_output_recorder():
    # transformers>=5 removed `OutputRecorder` and ignores `_can_record_outputs`, but remote code
    # (e.g. MiniMax-M2 modeling_minimax_m2.py) still imports it. Provide a no-op placeholder.
    from dataclasses import dataclass
    from transformers.utils import generic

    if hasattr(generic, 'OutputRecorder'):
        return

    @dataclass
    class OutputRecorder:
        target_class: type
        index: Optional[int] = None
        layer_name: Optional[str] = None

    generic.OutputRecorder = OutputRecorder


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    from torch.utils import cpp_extension

    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def _use_accuracy_compatible_enabled():
    """Whether ``use_accuracy_compatible`` is on, resolved at *import* time.

    ``_patch_mcore_bridge`` runs while ``swift.megatron`` is being imported, which is
    before ``MegatronArguments.__post_init__`` sets the ``USE_ACCURACY_COMPATIBLE`` env
    var. But the flag is already present in ``sys.argv`` at that point (the yaml
    ``use_accuracy_compatible: true`` is expanded by ``parse_yaml_args`` into
    ``--use_accuracy_compatible True`` and handed to the child process), so read it
    from argv here. Fall back to the env var for cases where it is set up front.
    """
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == '--use_accuracy_compatible':
            if i + 1 < len(argv):
                return str(argv[i + 1]).strip().lower() in ('1', 'true', 'yes')
            return True  # bare flag
        if arg.startswith('--use_accuracy_compatible='):
            return arg.split('=', 1)[1].strip().lower() in ('1', 'true', 'yes')
    return os.environ.get('USE_ACCURACY_COMPATIBLE', '0') == '1'


def _patch_mcore_bridge_disable_te():
    """Alignment: disable TransformerEngine at the layer-spec level (ex-ms-swift 175dd87).

    These three behaviors used to live directly in ms-swift
    ``model/{register,gpt_bridge,model_config}.py`` (commit 175dd87 "disabled TE").
    After the model/bridge/config layer was extracted into the ``mcore_bridge`` pip
    package they can no longer be committed as ms-swift source edits, and editing
    site-packages is not reproducible. We therefore re-apply them here as runtime
    monkeypatches on the imported mcore_bridge modules, so the change stays in
    versioned ms-swift source and survives a mcore_bridge reinstall.

    TE stays importable (``HAVE_TE`` is left True); we only avoid *building* TE
    layers by forcing the local layer spec. The router-gating GEMM alignment is
    handled separately by ``use_accuracy_compatible``.
    """
    # 1) force the local (non-TE) layer spec: use_transformer_engine=False for both
    #    the decoder block spec and the MTP block spec.
    import mcore_bridge.model.register as mcb_register

    def _force_local_spec(orig):

        def wrapper(*args, **kwargs):
            kwargs['use_transformer_engine'] = False
            return orig(*args, **kwargs)

        return wrapper

    mcb_register.get_gpt_decoder_block_spec = _force_local_spec(mcb_register.get_gpt_decoder_block_spec)
    mcb_register.get_gpt_mtp_block_spec = _force_local_spec(mcb_register.get_gpt_mtp_block_spec)

    # 1b) DSA is swapped in after the decoder spec via ModelLoader._replace_spec_dsa,
    # which called _get_backend_spec_provider (TESpecProvider). That left TELinear /
    # TENorm on DSA indexer + MLA while PaddleFleet HAVE_TE is False. E-259: remaining
    # Torch DSA TE contaminated post_attn_norm. Force LocalSpecProvider instead.
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as _eav

    def _local_backend_spec_provider(config):
        from megatron.core.models.backends import LocalSpecProvider
        return LocalSpecProvider()

    _eav._get_backend_spec_provider = _local_backend_spec_provider

    # 2) persist_layer_norm=False on the model config (dataclass default is baked into
    #    __init__, so flip it on the instance via __post_init__).
    from mcore_bridge.config.model_config import ModelConfig as McbModelConfig

    if not getattr(McbModelConfig.__post_init__, '_align_no_persist_ln', False):
        _origin_post_init = McbModelConfig.__post_init__

        def _post_init(self, *args, **kwargs):
            _origin_post_init(self, *args, **kwargs)
            self.persist_layer_norm = False

        _post_init._align_no_persist_ln = True
        McbModelConfig.__post_init__ = _post_init

    # 3) local-spec input-layernorm key mapping: with the local spec the input
    #    layernorm is a standalone `input_layernorm` module, not folded into
    #    `linear_qkv.layer_norm_weight`.
    from mcore_bridge.bridge.gpt_bridge import GPTBridge as McbGPTBridge

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx, to_mcore):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if self.config.multi_latent_attention:
            hf_state_dict.update(
                self._set_mla_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                 to_mcore)
        else:
            hf_state_dict.update(
                self._set_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                 to_mcore)
        return hf_state_dict

    McbGPTBridge._set_layer_attn = _set_layer_attn

    # 4) local-spec dense-MLP norm key mapping: with the local (non-TE) spec the
    #    dense MLP is `ColumnParallelLinear` + separate `pre_mlp_layernorm`
    #    (no fused `linear_fc1.layer_norm_weight`); route that key accordingly.
    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx, to_mcore, is_mtp=False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        is_moe = True if mg_mlp is not None and hasattr(mg_mlp, 'experts') else False
        if not to_mcore:
            is_moe = torch.tensor([is_moe], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_moe, group=self.pp_group)
        if is_moe:
            hf_state_dict.update(
                self._set_moe_state(
                    mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore, is_mtp=is_mtp))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict,
                                 self.hf_post_attention_layernorm_key, to_mcore)
        else:
            hf_state_dict.update(
                self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore))
            mg_fc1 = None if mg_layer is None else getattr(getattr(mg_layer, 'mlp', None), 'linear_fc1', None)
            fused_norm_weight = getattr(mg_fc1, 'layer_norm_weight', None)
            if fused_norm_weight is None:
                self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict,
                                     self.hf_post_attention_layernorm_key, to_mcore)
            else:
                self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                                     self.hf_post_attention_layernorm_key, to_mcore)
        return hf_state_dict

    McbGPTBridge._set_layer_mlp = _set_layer_mlp
    logger.info(
        'mcore_bridge patched for TE-off alignment (local spec, persist_layer_norm=False, input_layernorm+mlp-norm map)'
    )


def _patch_mcore_bridge():
    require_version(
        'mcore-bridge>=1.4.0',
        'please install mcore-bridge via `pip install mcore-bridge -U`',
    )
    import mcore_bridge
    from mcore_bridge import GPTBridge
    from mcore_bridge.model.register import ModelLoader
    logger.info(f"mcore_bridge.__version__: {mcore_bridge.__version__}")
    if _use_accuracy_compatible_enabled():
        _patch_mcore_bridge_disable_te()
    if not getattr(ModelLoader._replace_spec_dsa, '_swift_norm_accuracy_patch', False):
        origin_replace_spec_dsa = ModelLoader._replace_spec_dsa

        def replace_spec_dsa(self, layer_spec):
            origin_replace_spec_dsa(self, layer_spec)
            if not getattr(self.config, 'norm_accuracy_compatible', False):
                return
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            dsa_spec = layer_spec.submodules.self_attention
            dsa_spec.submodules.q_layernorm = WrappedTorchNorm
            dsa_spec.submodules.kv_layernorm = WrappedTorchNorm

        replace_spec_dsa._swift_norm_accuracy_patch = True
        ModelLoader._replace_spec_dsa = replace_spec_dsa
    origin_save_weights = GPTBridge.save_weights

    def save_weights(
        self,
        mg_models,
        output_dir: str,
        peft_format: bool = False,
        max_shard_size: str = '5GB',
        args=None,
        processor=None,
    ) -> None:
        origin_save_weights(
            self,
            mg_models,
            output_dir,
            peft_format=peft_format,
            max_shard_size=max_shard_size,
        )
        if processor is None or args is None:
            return
        hf_config = self.config.hf_config
        hf_config = deepcopy(hf_config)
        if is_master() and not hasattr(self, 'hf_model'):
            if hasattr(self, 'get_hf_meta_model'):
                self.hf_model = self.get_hf_meta_model()
                self.hf_model.model_meta = processor.model_meta
                self.hf_model.model_info = processor.model_info
            else:
                with torch.device('meta'), disable_safe_ddp_context_use_barrier():
                    self.hf_model = get_model_processor(
                        args.model_dir,
                        model_type=args.model_type,
                        return_dummy_model=True,
                        processor_id_or_path=_get_save_processor_id(args),
                    )[0]

        if is_master():
            if peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if self.config.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if self.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules,
                    )
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                config = self.config
                llm_config = HfConfigFactory.get_text_config(hf_config)
                if config.mtp_num_layers:
                    for key in ['num_nextn_predict_layers', 'mtp_num_hidden_layers']:
                        if hasattr(llm_config, key):
                            setattr(llm_config, key, config.mtp_num_layers)
                            break
                    else:
                        llm_config.num_nextn_predict_layers = config.mtp_num_layers
                HfConfigFactory.del_config_attr(hf_config, 'quantization_config')
                expert_dtype = None
                if (config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param):
                    from transformers.utils.quantization_config import FineGrainedFP8Config

                    modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                    if hasattr(self, '_fp8_skip_modules'):
                        modules_to_not_convert = (modules_to_not_convert or []) + list(self._fp8_skip_modules)
                    hf_config.quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
                    expert_dtype = 'fp8'
                if args.model_type == 'deepseek_v4':
                    HfConfigFactory.set_config_attr(hf_config, 'expert_dtype', expert_dtype)
                hf_config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=hf_config)
                    except FileNotFoundError as e:
                        logger.error(f"custom_object_save Error: {e}")
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files,
                )
            logger.info(f"Successfully saved `safetensors` model weights in `{output_dir}`.")
        dist.barrier()  # Ensure all weights are saved completely

    GPTBridge.save_weights = save_weights


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    _patch_transformers_output_recorder()
    _patch_mcore_bridge()
    _patch__batched_p2p_ops()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass
    import megatron.core

    logger.info(f"megatron.core.__version__: {megatron.core.__version__}")
