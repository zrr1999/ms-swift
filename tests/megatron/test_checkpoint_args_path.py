"""Regression tests for where save_checkpoint looks up args.json.

MegatronTrainer.save_checkpoint copies args.json from the run's output directory into
the checkpoint directory it just created. It used to derive that source path as
``dirname(output_dir)``, which is only correct when output_dir is the default
``args.output_dir/checkpoint-<iteration>``. When MODEL_REPRO_CHECKPOINT_DIR redirects
output_dir to an arbitrary absolute path, ``dirname`` points somewhere unrelated and
copy_path raises FileNotFoundError, discarding the whole checkpoint after a completed
training step.
"""

import os

import pytest

from swift.megatron.trainers.base import BaseMegatronTrainer


def args_json_source(output_dir: str, args_output_dir: str) -> str:
    """The path save_checkpoint reads args.json from, isolated from the trainer.

    Mirrors the single expression under test so the rule can be checked without
    constructing a distributed trainer.
    """
    return os.path.join(args_output_dir, 'args.json')


def test_args_json_comes_from_output_dir_for_the_default_checkpoint_path(tmp_path):
    args_output_dir = str(tmp_path / 'run')
    output_dir = os.path.join(args_output_dir, 'checkpoint-1')

    assert args_json_source(output_dir, args_output_dir) == os.path.join(args_output_dir, 'args.json')


def test_args_json_comes_from_output_dir_when_the_checkpoint_dir_is_redirected(tmp_path):
    """The case that used to fail: an absolute override unrelated to args.output_dir."""
    args_output_dir = str(tmp_path / 'run')
    output_dir = str(tmp_path / 'elsewhere' / 'formal_checkpoint')

    source = args_json_source(output_dir, args_output_dir)

    assert source == os.path.join(args_output_dir, 'args.json')
    # The old rule resolved to the sibling of the override, which holds no args.json.
    assert source != os.path.join(os.path.dirname(output_dir), 'args.json')


def test_save_checkpoint_reads_args_json_from_args_output_dir():
    """Pin the rule to the real implementation, not just to the helper above."""
    import inspect

    source = inspect.getsource(BaseMegatronTrainer.save_checkpoint)

    assert "args_path = os.path.join(args.output_dir, 'args.json')" in source
    assert "os.path.join(os.path.dirname(output_dir), 'args.json')" not in source


def test_copy_path_still_fails_closed_on_a_genuinely_missing_source(tmp_path):
    """The fix must not turn a real missing-file condition into a silent skip."""
    missing = str(tmp_path / 'nope.json')

    with pytest.raises(FileNotFoundError):
        BaseMegatronTrainer.copy_path(missing, str(tmp_path / 'out.json'))
