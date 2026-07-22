from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

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
    monkeypatch.setattr(
        utils, 'fields', lambda _: [SimpleNamespace(name='mtp_num_layers'), SimpleNamespace(name='num_moe_experts')])


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
