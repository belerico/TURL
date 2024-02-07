import copy
import glob
import json
import os
import pickle
import re
import shutil
import warnings
from collections import OrderedDict
from functools import partial
from itertools import repeat
from logging import Logger
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def create_ent_embedding(data_dir, ent_vocab, origin_embed):
    with open(os.path.join(data_dir, "entity_embedding_tinybert_312.pkl"), "rb") as f:
        ent_embed = pickle.load(f)
    for wiki_id in tqdm(ent_vocab):
        ent_id = ent_vocab[wiki_id]
        ent_embedding = ent_embed[str(wiki_id)]
        origin_embed[ent_id] = ent_embedding
    return origin_embed


def create_header_embedding(data_dir, header_vocab, origin_embed, is_bert=False):
    with open(
        os.path.join(data_dir, "header_embedding_312_bert.pkl" if is_bert else "header_embedding_312.pkl"), "rb"
    ) as f:
        header_embed = pickle.load(f)
    for header_id in header_vocab:
        origin_embed[header_id] = header_embed[header_vocab[header_id]]
    return origin_embed


RESERVED_ENT_VOCAB = {
    0: {"wiki_id": "[PAD]", "wiki_title": "[PAD]", "count": -1, "mid": -1},
    1: {"wiki_id": "[ENT_MASK]", "wiki_title": "[ENT_MASK]", "count": -1, "mid": -1},
    2: {"wiki_id": "[PG_ENT_MASK]", "wiki_title": "[PG_ENT_MASK]", "count": -1, "mid": -1},
    3: {"wiki_id": "[CORE_ENT_MASK]", "wiki_title": "[CORE_ENT_MASK]", "count": -1, "mid": -1},
}
RESERVED_ENT_VOCAB_NUM = len(RESERVED_ENT_VOCAB)


def load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=1):
    entity_vocab = copy.deepcopy(RESERVED_ENT_VOCAB)
    bad_title = 0
    few_entity = 0
    with open(os.path.join(data_dir, "entity_vocab.txt"), "r", encoding="utf-8") as f:
        for line in f:
            _, entity_id, entity_title, entity_mid, count = line.strip().split("\t")
            if ignore_bad_title and entity_title == "":
                bad_title += 1
            elif int(count) < min_ent_count:
                few_entity += 1
            else:
                entity_vocab[len(entity_vocab)] = {
                    "wiki_id": int(entity_id),
                    "wiki_title": entity_title,
                    "mid": entity_mid,
                    "count": int(count),
                }
    print(
        "total number of entity: %d\nremove because of empty title: %d\nremove because count<%d: %d"
        % (len(entity_vocab), bad_title, min_ent_count, few_entity)
    )
    return entity_vocab


def generate_vocab_distribution(entity_vocab: Dict[int, Any]) -> np.ndarray:
    distribution = np.zeros(len(entity_vocab))
    for i, item in entity_vocab.items():
        if i in RESERVED_ENT_VOCAB:
            distribution[i] = 2
        else:
            distribution[i] = int(item["count"])
    distribution = np.log10(distribution)
    distribution /= np.sum(distribution)
    return distribution


def load_type_vocab(data_dir):
    type_vocab = {}
    with open(os.path.join(data_dir, "type_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split("\t")
            type_vocab[t] = int(index)
    return type_vocab


def load_relation_vocab(data_dir):
    relation_vocab = {}
    with open(os.path.join(data_dir, "relation_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split("\t")
            relation_vocab[t] = int(index)
    return relation_vocab


def load_dbpedia_type_vocab(data_dir):
    type_vocab = {}
    with open(os.path.join(data_dir, "dbpedia_type_vocab.txt"), "r") as f:
        for line in f:
            index, t = line.strip().split("\t")
            type_vocab[t] = int(index)
    return type_vocab


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def _get_linear_schedule_with_warmup_from_init_to_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, init_lr: float, final_lr: float
):
    if current_step < num_warmup_steps:
        return (final_lr - init_lr) / float(num_warmup_steps) * float(current_step) + init_lr
    return max(
        0.0,
        final_lr * (1 - float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))),
    )


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, init_lrs=None, lrs_after_warmup=None
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if lrs_after_warmup is None:
        lr_lambdas = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        if init_lrs is None:
            init_lrs_per_group = [copy.deepcopy(param_group["lr"]) for param_group in optimizer.param_groups]
        elif not isinstance(init_lrs, (list, tuple)):
            init_lrs_per_group = [init_lrs for _ in optimizer.param_groups]
        else:
            if len(init_lrs) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} initial learning rates, but got {len(init_lrs)}"
                )
            init_lrs_per_group = init_lrs
        if not isinstance(lrs_after_warmup, (list, tuple)):
            lrs_after_warmup = [lrs_after_warmup for _ in optimizer.param_groups]
        else:
            if len(lrs_after_warmup) != len(init_lrs_per_group):
                raise ValueError(
                    f"Expected {len(init_lrs_per_group)} lrs to reach after warmup, but got {len(lrs_after_warmup)}"
                )
        warnings.warn(
            "Setting the learning rate of every param group to 1.0. This is done to simplify the learning rate scheduling code.",
            UserWarning,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1.0
        lr_lambdas = [
            (
                partial(
                    _get_linear_schedule_with_warmup_from_init_to_warmup_lr_lambda,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    init_lr=init_lr,
                    final_lr=final_lr,
                )
            )
            for init_lr, final_lr in zip(init_lrs_per_group, lrs_after_warmup)
        ]
    return LambdaLR(optimizer, lr_lambdas, last_epoch)


def rotate_checkpoints(
    args,
    checkpoint_prefix,
    use_mtime=False,
    log_dir: str = None,
    checkpoint_dir: str = "checkpoints",
    logger: Logger = None,
):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir if log_dir is None else log_dir, checkpoint_dir, "{}-*".format(checkpoint_prefix))
    )
    if len(glob_checkpoints) == 0:
        warnings.warn(
            f"Found no checkpoints at {os.path.join(args.output_dir if log_dir is None else log_dir, checkpoint_dir)}, cannot rotate checkpoints"
        )
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        if logger is not None:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
