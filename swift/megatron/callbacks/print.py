# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
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
        for key, value in logs.items() if key == 'loss' or (key.startswith('mtp_') and key.endswith('_loss'))
    }
    return {'step': step, **raw_losses} if raw_losses else None


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.raw_loss_writer = None
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
            self.raw_loss_writer = JsonlWriter(raw_loss_path, write_on_rank='last')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

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
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            self.training_bar.write(str(logs))
