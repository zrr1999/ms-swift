import torch

from swift.megatron.callbacks.print import (
    machine_loss_payload,
    normalized_device,
    normalized_dtype,
    raw_loss_event,
)
from swift.megatron.trainers.trainer import MegatronTrainer, project_owning_loader_semantics


def test_raw_loss_event_preserves_unrounded_values_and_step():
    logs = {
        "loss": 12.410510059999999,
        "mtp_0_loss": 13.367947579999999,
        "eval_loss": 99.0,
    }
    assert raw_loss_event(1, logs) == {
        "step": 1,
        "loss": 12.410510059999999,
        "mtp_0_loss": 13.367947579999999,
    }


def test_raw_loss_event_omits_non_loss_metrics():
    assert raw_loss_event(3, {"grad_norm": 1.0, "learning_rate": 1e-6}) is None


def test_machine_loss_payload_exposes_unrounded_losses_gate_field():
    events = [
        {"step": 1, "loss": 11.821867942810059, "mtp_1_loss": 12.910821914672852},
        {"step": 2, "loss": 11.795001983642578, "mtp_1_loss": 12.899291038513184},
    ]
    payload = machine_loss_payload(events, raw_path="results/torch/raw_loss.jsonl")

    assert payload["losses"] == [11.821867942810059, 11.795001983642578]
    assert payload["steps"] == [1, 2]
    assert payload["events"] is events
    assert payload["framework"] == "torch"
    assert payload["raw"] is True


def test_machine_loss_payload_losses_excludes_mtp_and_survives_missing_loss():
    payload = machine_loss_payload([{"step": 1, "mtp_1_loss": 12.5}, {"step": 2, "loss": 9.5}])
    assert payload["losses"] == [9.5]
    assert payload["event_count"] == 2


def test_normalized_dtype_strips_torch_prefix_for_bench_alias():
    assert normalized_dtype(torch.bfloat16) == "bfloat16"
    assert normalized_dtype("torch.bfloat16") == "bfloat16"
    assert normalized_dtype("bfloat16") == "bfloat16"


def test_normalized_device_reports_device_class_not_model_name():
    device = normalized_device()
    assert device in {"cuda", "cpu"}
    assert "H800" not in device


def test_owning_loader_projection_removes_sp_padding_and_reverses_label_roll():
    input_values = list(range(57)) + [154820]
    original_labels = [-100] * 13 + list(range(44)) + [-100]
    shifted_labels = original_labels[1:] + original_labels[:1]
    semantic_input, semantic_labels, semantic_mask = project_owning_loader_semantics(
        input_values, shifted_labels, semantic_length=57, labels_were_shifted=True)
    assert semantic_input == list(range(57))
    assert semantic_labels == original_labels[:57]
    assert sum(semantic_mask) == 44


def test_owning_loader_projection_rejects_invalid_semantic_length():
    try:
        project_owning_loader_semantics([1, 2], [-100, 2], semantic_length=3)
    except ValueError as exc:
        assert "invalid owning-loader semantic length" in str(exc)
    else:
        raise AssertionError("invalid semantic length was accepted")


def test_parameter_record_preserves_orientation_and_signed_zero():
    tensor = torch.tensor([[0.0, -0.0], [1.0, 2.0]], dtype=torch.float32)
    record = MegatronTrainer._parameter_record(tensor)
    assert record["shape"] == [2, 2]
    assert record["dtype"] == "torch.float32"
    assert record["positive_zero_count"] == 1
    assert record["negative_zero_count"] == 1
    assert record["sha256"] != record["transpose_sha256"]


def test_layer0_fine_forward_specs_are_explicit_and_fail_closed():
    assert MegatronTrainer._forward_contract_specs("coarse") is None
    specs = MegatronTrainer._forward_contract_specs("layer0_fine")
    assert len(specs) == 13
    assert specs["decoder.layers.0.input_layernorm"] == "layer0_input_rmsnorm_output"
    assert specs["decoder.layers.0.self_attention.linear_q_down_proj"] == "layer0_q_down_projection_output"
    assert specs["decoder.layers.0"] == "base_transformer_layer_0_output"
    try:
        MegatronTrainer._forward_contract_specs("unknown")
    except ValueError as exc:
        assert "unsupported MODEL_REPRO_FORWARD_BOUNDARY_SET" in str(exc)
    else:
        raise AssertionError("unknown forward boundary set was accepted")


def test_first_tensor_prefers_first_tensor_in_nested_module_output():
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    assert MegatronTrainer._first_tensor((None, {"output": first}, second)) is first
