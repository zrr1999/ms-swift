import inspect

import torch
from transformers import PretrainedConfig
from types import SimpleNamespace

from swift.megatron.init import _get_save_processor_id, _patch_mcore_bridge_disable_te
from swift.megatron.model import utils


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
    assert _get_save_processor_id(
        SimpleNamespace(model_dir='/weights', tokenizer_name_or_path='/tokenizer')
    ) == '/tokenizer'
    assert _get_save_processor_id(
        SimpleNamespace(model_dir='/weights', tokenizer_name_or_path=None)
    ) == '/weights'


def test_get_mcore_model_config_reads_mtp_num_layers_from_hf(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_nextn_predict_layers=1)

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert config.kwargs['mtp_num_layers'] == 1


def test_get_mcore_model_config_reads_mtp_num_hidden_layers(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(text_config=PretrainedConfig(mtp_num_hidden_layers=1))

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert config.kwargs['mtp_num_layers'] == 1


def test_get_mcore_model_config_prefers_n_routed_experts(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_experts=256, n_routed_experts=16)

    config = utils.get_mcore_model_config(_make_args(), hf_config)

    assert config.kwargs['num_moe_experts'] == 16


def test_get_mcore_model_config_keeps_explicit_mtp_num_layers(monkeypatch):
    _patch_model_config(monkeypatch)
    hf_config = PretrainedConfig(num_nextn_predict_layers=1)

    config = utils.get_mcore_model_config(_make_args(mtp_num_layers=2), hf_config)

    assert config.kwargs['mtp_num_layers'] == 2


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
    assert "fused_norm_weight is None" in source
    assert "pre_mlp_layernorm.weight" in source
    assert "mlp.linear_fc1.layer_norm_weight" in source


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
