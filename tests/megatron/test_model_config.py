import inspect
import math
import torch
from transformers import PretrainedConfig
from types import SimpleNamespace

from swift.megatron.init import _get_save_processor_id, _patch_mcore_bridge_disable_te
from swift.megatron.model import utils
from swift.megatron.utils.utils import get_padding_to


class _ModelConfigStub:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.attention_backend = SimpleNamespace(name='unfused')
        self.experimental_attention_variant = 'dsa'


def _make_args(mtp_num_layers=None):
    return SimpleNamespace(
        megatron_model_meta=SimpleNamespace(model_type='gpt'),
        mtp_num_layers=mtp_num_layers,
        task_type='causal_lm',
        torch_dtype=torch.bfloat16,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        fp4_param_gather=False,
        fp8_param_gather=False,
        moe_grouped_gemm=False,
        router_replay_mode='disabled',
        megatron_extra_kwargs=None,
        padding_free=False,
    )


def _patch_model_config(monkeypatch):
    monkeypatch.setattr(utils, 'ModelConfig', _ModelConfigStub)
    monkeypatch.setattr(utils, 'fields',
                        lambda _: [SimpleNamespace(name='mtp_num_layers'),
                                   SimpleNamespace(name='num_moe_experts')])


def test_save_processor_prefers_independent_tokenizer_source():
    assert _get_save_processor_id(SimpleNamespace(model_dir='/weights',
                                                  tokenizer_name_or_path='/tokenizer')) == '/tokenizer'
    assert _get_save_processor_id(SimpleNamespace(model_dir='/weights', tokenizer_name_or_path=None)) == '/weights'


def test_get_mcore_model_config_propagates_accuracy_mode(monkeypatch):
    # Preserve the real inherited dataclass fields while avoiding model construction.
    actual_fields = utils.fields(utils.ModelConfig)
    assert 'use_accuracy_compatible' in {field.name for field in actual_fields}
    monkeypatch.setattr(utils, 'ModelConfig', _ModelConfigStub)
    monkeypatch.setattr(utils, 'fields', lambda _: actual_fields)
    for enabled in (False, True, False):
        args = _make_args()
        args.use_accuracy_compatible = enabled
        monkeypatch.setenv('USE_ACCURACY_COMPATIBLE', str(int(not enabled)))
        config = utils.get_mcore_model_config(args, PretrainedConfig())
        assert config.kwargs['use_accuracy_compatible'] is enabled


def test_get_mcore_model_config_does_not_enable_mtp_from_checkpoint(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_nextn_predict_layers=1)

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert not config.kwargs.get('mtp_num_layers')


def test_glm52_loss_sum_contract_keeps_other_models_default(monkeypatch):
    _patch_model_config(monkeypatch)
    for model_type in ('glm_moe_dsa', 'glm4_moe', 'minimax_m2'):
        hf_config = PretrainedConfig(model_type=model_type)
        config = utils.get_mcore_model_config(_make_args(), hf_config)
        assert config.kwargs.get('accuracy_compatible_loss_sum_dtype',
                                 'float64') == ('float32' if model_type == 'glm_moe_dsa' else 'float64')


def test_loss_sum_contract_rejects_unsupported_dtype(monkeypatch):
    _patch_model_config(monkeypatch)
    args = _make_args()
    args.megatron_extra_kwargs = {'accuracy_compatible_loss_sum_dtype': 'bfloat16'}
    try:
        utils.get_mcore_model_config(args, PretrainedConfig())
    except ValueError as error:
        assert 'accuracy_compatible_loss_sum_dtype' in str(error)
    else:
        raise AssertionError('BF16 loss accumulation must fail before constructing the model')


def test_get_mcore_model_config_does_not_enable_mtp_from_nested_checkpoint(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(text_config=PretrainedConfig(mtp_num_hidden_layers=1))

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert not config.kwargs.get('mtp_num_layers')


def test_get_mcore_model_config_prefers_n_routed_experts(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_experts=256, n_routed_experts=16)

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert config.kwargs['num_moe_experts'] == 16


def test_get_mcore_model_config_keeps_explicit_mtp_num_layers(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_nextn_predict_layers=1)

    for depth in (0, 1, 2):
        config = utils.get_mcore_model_config(_make_args(mtp_num_layers=depth), hf_config)
        assert config.kwargs['mtp_num_layers'] == depth


def test_get_padding_to_sequence_parallel_uses_tp_times_two():
    args = SimpleNamespace(
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=1,
        fp8_recipe='delayed',
        fp8_format=None,
        fp8=None,
        fp4_format=None,
        fp4=None,
        attention_backend='unfused',
    )
    assert get_padding_to(args) == 4
    seq_len = 57
    assert math.ceil(seq_len / 4) * 4 == 60
    assert math.ceil(seq_len / 2) * 2 == 58


def test_dsa_index_share_allows_recompute_none():
    config = SimpleNamespace(
        experimental_attention_variant='dsa',
        dsa_indexer_topk_freq=4,
        recompute_granularity='none',
    )

    utils._check_dsa_index_share_recompute(config)


def test_dsa_backend_forced_to_local_spec_when_accuracy_compatible(monkeypatch):
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as eav

    import swift.megatron.init as init

    monkeypatch.setattr(init, '_use_accuracy_compatible_enabled', lambda: True)
    init._patch_mcore_bridge_disable_te()
    provider = eav._get_backend_spec_provider(SimpleNamespace())
    assert isinstance(provider, LocalSpecProvider)
    assert hasattr(provider, 'linear')
    assert provider.linear() is not provider.column_parallel_linear()


def test_local_spec_mlp_norm_maps_pre_mlp_layernorm_when_unfused():
    source = inspect.getsource(_patch_mcore_bridge_disable_te)
    assert 'fused_norm_weight is None' in source
    assert 'pre_mlp_layernorm.weight' in source
    assert 'mlp.linear_fc1.layer_norm_weight' in source


def test_dsa_index_share_rejects_selective_recompute():
    config = SimpleNamespace(
        experimental_attention_variant='dsa',
        dsa_indexer_topk_freq=4,
        recompute_granularity='selective',
    )

    try:
        utils._check_dsa_index_share_recompute(config)
    except ValueError as error:
        assert 'Set recompute_granularity=none' in str(error)
    else:
        raise AssertionError('expected DSA index sharing with selective recompute to fail closed')
