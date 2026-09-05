import pytest
import torch
from datasets import Dataset
from types import SimpleNamespace

from swift.dataset import validate_pretokenized_dataset
from swift.megatron.trainers import utils as trainer_utils
from swift.model.register import ModelLoader


def test_validate_pretokenized_dataset_accepts_fixed_tokens():
    dataset = Dataset.from_dict({
        'input_ids': [[154820, 42, 42, 17, 99, 42, 8]],
        'labels': [[42, 42, 17, 99, 42, 8, 3]],
        'position_ids': [[0, 1, 2, 3, 4, 5, 6]],
        'lengths': [7],
    })

    validate_pretokenized_dataset(dataset, max_length=7)


def test_validate_pretokenized_dataset_rejects_missing_labels():
    dataset = Dataset.from_dict({
        'input_ids': [[1, 2]],
        'position_ids': [[0, 1]],
        'lengths': [2],
    })

    with pytest.raises(ValueError, match='missing columns'):
        validate_pretokenized_dataset(dataset, max_length=2)


def test_pretokenized_labels_are_not_shifted(monkeypatch):
    labels = torch.tensor([[42, 42, 17, 99, 42, 8, 3]])
    data = {'input_ids': torch.zeros_like(labels), 'labels': labels.clone()}
    args = SimpleNamespace(
        task_type='causal_lm',
        pretokenized_dataset=True,
        pipeline_model_parallel_size=1,
    )
    monkeypatch.setattr(trainer_utils, 'get_current_device', lambda: 'cpu')
    monkeypatch.setattr(trainer_utils, 'to_device', lambda value, *_args, **_kwargs: value)

    batch = trainer_utils.get_batch_on_this_pp_rank(args, data)

    assert torch.equal(batch['labels'], labels)


def test_model_loader_uses_independent_processor_path():
    requested = []

    class FakeTokenizer:

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            requested.append((path, kwargs))
            return object()

    loader = ModelLoader.__new__(ModelLoader)
    loader.processor_id_or_path = '/tokenizer-only'
    loader.auto_tokenizer_cls = FakeTokenizer
    loader.default_trust_remote_code = True

    loader.get_processor('/weights-only', config=None)

    assert requested == [('/tokenizer-only', {'trust_remote_code': True})]
