# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


def raw_loss_event(step, logs):
    """Return an unrounded training-loss event, excluding evaluation metrics."""
    raw_losses = {
        key: value
        for key, value in logs.items()
        if key == 'loss' or (key.startswith('mtp_') and key.endswith('_loss'))
    }
    return {'step': step, **raw_losses} if raw_losses else None


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_device():
    """Return the device class the benchmark checker expects, not the GPU model name."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def normalized_dtype(value):
    """Return the bench dtype alias; a bare ``torch.bfloat16`` string is rejected."""
    text = str(value or '').strip()
    prefix = 'torch.'
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text.lower()


def machine_loss_payload(events, raw_path=None, source_sha256=None):
    """Return the machine loss artifact.

    ``losses`` is the benchmark gate field: an unrounded main-loss series with one
    entry per recorded step. ``events`` keeps the per-step diagnostic detail.
    """
    return {
        'schema': 'glm52-machine-loss/v1',
        'framework': 'torch',
        'raw': True,
        'owning_cli_exit_code': 0,
        'losses': [event['loss'] for event in events if 'loss' in event],
        'event_count': len(events),
        'steps': [event['step'] for event in events],
        'events': events,
        'source': raw_path,
        'source_sha256': source_sha256,
    }


def _write_json(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + '\n')


def model_repro_environment(args):
    """Return the formal run-local environment receipt after model loading."""
    config_path = os.environ.get('MODEL_REPRO_MODEL_CONFIG_PATH')
    return {
        'schema': 'glm52-environment/v1',
        'framework': 'torch',
        'framework_version': torch.__version__,
        'python_version': platform.python_version(),
        'device': normalized_device(),
        'device_name': torch.cuda.get_device_name(torch.cuda.current_device()),
        'dtype': normalized_dtype(getattr(args, 'torch_dtype', 'bfloat16')),
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'nccl': list(torch.cuda.nccl.version()),
        'deterministic': {
            'algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'nccl_algo': os.environ.get('NCCL_ALGO'),
        },
        'model_id': os.environ.get('MODEL_REPRO_MODEL_ID'),
        'revision': os.environ.get('MODEL_REPRO_MODEL_REVISION'),
        'model_config_sha256': _sha256_file(config_path) if config_path else None,
        'weights_loaded': True,
        'model_source': str(getattr(args, 'model', '')),
        'world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
    }


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.raw_loss_writer = None
        self.raw_loss_events = []
        self.is_write_rank = is_last_rank()

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        logging_path = os.path.join(self.args.output_dir, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')
        raw_loss_path = os.environ.get('MODEL_REPRO_RAW_LOSS_PATH')
        if raw_loss_path:
            logger.info(f'raw_loss_path: {raw_loss_path}')
            if self.is_write_rank and Path(raw_loss_path).exists():
                Path(raw_loss_path).unlink()
            self.raw_loss_writer = JsonlWriter(raw_loss_path, write_on_rank='last')
        env_path = os.environ.get('MODEL_REPRO_ENV_PATH')
        if env_path and self.is_write_rank:
            _write_json(env_path, model_repro_environment(self.args))

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None
        loss_path = os.environ.get('MODEL_REPRO_LOSS_PATH')
        if loss_path and self.is_write_rank:
            raw_path = os.environ.get('MODEL_REPRO_RAW_LOSS_PATH')
            payload = machine_loss_payload(
                self.raw_loss_events,
                raw_path=raw_path,
                source_sha256=_sha256_file(raw_path) if raw_path and Path(raw_path).is_file() else None,
            )
            _write_json(loss_path, payload)

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        elapsed = time.time() - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.start_step
        train_speed = elapsed / n_steps if n_steps > 0 else 0.0
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        raw_event = raw_loss_event(state.iteration, logs)
        if self.raw_loss_writer is not None and raw_event is not None:
            self.raw_loss_writer.append(raw_event)
        if raw_event is not None and self.is_write_rank:
            self.raw_loss_events.append(raw_event)
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            self.training_bar.write(str(logs))
