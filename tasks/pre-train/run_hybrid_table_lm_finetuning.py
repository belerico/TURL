# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, annotations, division, print_function

import argparse
import glob
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lightning import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME

from src.data_loader.data_loaders import EntityTableLoader, WikiEntityTableDataset
from src.data_loader.hybrid_data_loaders import HybridTableLoader, WikiHybridTableDataset
from src.model.configuration import TableConfig
from src.model.metric import accuracy, mean_average_precision, mean_rank, top_k_acc
from src.model.model import HybridTableMaskedLM
from src.model.optim import DenseSparseAdam
from src.utils.util import (
    create_ent_embedding,
    generate_vocab_distribution,
    get_linear_schedule_with_warmup,
    load_entity_vocab,
    rotate_checkpoints,
)

logger = logging.getLogger(__name__)


def train(
    args: argparse.Namespace,
    config: TableConfig,
    train_dataset: Dataset,
    model: HybridTableMaskedLM,
    tb_logger: TensorBoardLogger,
    eval_dataset: Dataset | None = None,
    sample_distribution: np.ndarray | None = None,
):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = HybridTableLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        max_entity_candidate=args.max_entity_candidate,
        num_workers=0,
        mlm_probability=args.mlm_probability,
        ent_mlm_probability=args.ent_mlm_probability,
        mall_probability=args.mall_probability,
        is_train=True,
        sample_distribution=sample_distribution,
        use_cand=args.use_cand,
        random_sample=args.random_sample,
        use_visibility=False if args.no_visibility else True,
    )

    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // steps_per_epoch + 1
    else:
        t_total = steps_per_epoch * args.num_train_epochs

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = DenseSparseAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * steps_per_epoch,
        num_training_steps=t_total,
        init_lrs=args.learning_rate,
        lrs_after_warmup=args.scaled_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    if args.resume != "":
        state_for_resume = torch.load(os.path.join(args.resume, "state_for_resume.bin"))
        scheduler.load_state_dict(state_for_resume["scheduler"])
        optimizer.load_state_dict(state_for_resume["optimizer"])
        scaler.load_state_dict(state_for_resume["scaler"])
        global_step = state_for_resume["global_step"]
        logger.info("resume from %s" % args.resume)
    else:
        global_step = 0

    # Distributed training (should be after apex fp16 initialization)
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
        )

    # Train!
    if args.local_rank in {-1, 0}:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (dist.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    tr_loss, logging_loss = 0.0, 0.0
    tok_tr_loss, tok_logging_loss, ent_tr_loss, ent_logging_loss = 0.0, 0.0, 0.0, 0.0

    model.train()
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)

    # Log first learning rate
    if args.local_rank in {-1, 0} and args.logging_steps > 0:
        tb_logger.log_metrics({"train/lr": scheduler.get_last_lr()[0]}, global_step)

    # Train loop
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in {-1, 0})
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in {-1, 0})
        for step, batch in enumerate(epoch_iterator):
            is_accumulating = (step + 1) % args.gradient_accumulation_steps != 0

            (
                _,
                input_tok,
                input_tok_type,
                input_tok_pos,
                input_tok_labels,
                input_tok_mask,
                input_ent_text,
                input_ent_text_length,
                input_ent_mask_type,
                input_ent,
                input_ent_type,
                input_ent_labels,
                input_ent_mask,
                candidate_entity_set,
                _,
                exclusive_ent_mask,
                core_entity_mask,
            ) = batch
            input_tok = input_tok.to(args.device)
            input_tok_type = input_tok_type.to(args.device)
            input_tok_pos = input_tok_pos.to(args.device)
            input_tok_mask = input_tok_mask.to(args.device)
            input_tok_labels = input_tok_labels.to(args.device)
            input_ent_text = input_ent_text.to(args.device)
            input_ent_text_length = input_ent_text_length.to(args.device)
            input_ent_mask_type = input_ent_mask_type.to(args.device)
            input_ent = input_ent.to(args.device)
            input_ent_type = input_ent_type.to(args.device)
            input_ent_mask = input_ent_mask.to(args.device)
            input_ent_labels = input_ent_labels.to(args.device)
            candidate_entity_set = candidate_entity_set.to(args.device)
            core_entity_mask = core_entity_mask.to(args.device)
            if args.exclusive_ent == 0:  # no mask
                exclusive_ent_mask = None
            else:
                exclusive_ent_mask = exclusive_ent_mask.to(args.device)
                if args.exclusive_ent == 2:  # mask only core entity
                    exclusive_ent_mask += (~core_entity_mask[:, :, None]).long() * 1000

            # Do not sync gradients between processes if we are accumulating
            no_backward_sync_ctx = nullcontext()
            if is_accumulating and dist.is_available() and dist.is_initialized():
                no_backward_sync_ctx = model.no_sync()
            with no_backward_sync_ctx:
                with torch.autocast(
                    device_type="cuda" if str(args.device).startswith("cuda") else "cpu",
                    dtype=torch.float16,
                    enabled=args.fp16,
                ):
                    tok_outputs, ent_outputs = model(
                        input_tok,
                        input_tok_type,
                        input_tok_pos,
                        input_tok_mask,
                        input_ent_text,
                        input_ent_text_length,
                        input_ent_mask_type,
                        input_ent,
                        input_ent_type,
                        input_ent_mask,
                        candidate_entity_set,
                        input_tok_labels,
                        input_ent_labels,
                        exclusive_ent_mask,
                    )
                    tok_loss = tok_outputs[0]  # model outputs are always tuple in transformers (see doc)
                    ent_loss = ent_outputs[0]
                    loss = tok_loss + ent_loss

                # Loss reduction
                loss = loss.mean()
                tok_loss = tok_loss.mean()
                ent_loss = ent_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    tok_loss = tok_loss / args.gradient_accumulation_steps
                    ent_loss = ent_loss / args.gradient_accumulation_steps

                # Scaled loss backward for mixed precision training
                scaler.scale(loss).backward()

            tr_loss += loss.item()
            tok_tr_loss += tok_loss.item()
            ent_tr_loss += ent_loss.item()

            if not is_accumulating:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer and scheduler update
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Update global step
                global_step += 1

                # Log metrics
                if args.local_rank in {-1, 0} and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        results = evaluate(
                            args,
                            config,
                            eval_dataset,
                            getattr(model, "module", model),
                            prefix="checkpoints/checkpoint-{}".format(global_step),
                            sample_distribution=None,
                            log_dir=tb_logger.log_dir,
                        )

                        # Put model back to training mode
                        model.train()

                        # Log eval metrics with TensorBoard
                        for key, value in results.items():
                            tb_logger.log_metrics({"eval/{}".format(key): value}, global_step)

                    # Log training metrics with TensorBoard
                    tb_logger.log_metrics(
                        {
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/loss": (tr_loss - logging_loss) / args.logging_steps,
                            "train/tok_loss": (tok_tr_loss - tok_logging_loss) / args.logging_steps,
                            "train/ent_loss": (ent_tr_loss - ent_logging_loss) / args.logging_steps,
                            "train/grad_norm": grad_norm,
                        },
                        global_step,
                    )
                    logging_loss = tr_loss
                    tok_logging_loss = tok_tr_loss
                    ent_logging_loss = ent_tr_loss

                if args.local_rank in {-1, 0} and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(tb_logger.log_dir, "checkpoints", "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    model_to_save = getattr(model, "module", model)
                    model_to_save.save_pretrained(output_dir)

                    # Save CLI args
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))

                    # Save optimizer, scheduler and scaler
                    state_for_resume = {
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "global_step": global_step,
                    }
                    torch.save(state_for_resume, os.path.join(output_dir, "state_for_resume.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # Remove older checkpoints
                    rotate_checkpoints(args, "checkpoint", log_dir=tb_logger.log_dir)

                # Possibly wait for rank-0 to log metrics and save model
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    if args.local_rank in {-1, 0}:
        tb_logger.finalize("success")

    return global_step, tr_loss / max(global_step, 1)


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    config: TableConfig,
    eval_dataset: Dataset,
    model: HybridTableMaskedLM,
    prefix: str = "",
    sample_distribution: np.ndarray | None = None,
    log_dir: str | None = None,
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir if log_dir is None else log_dir
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

    if args.local_rank in {-1, 0}:
        os.makedirs(os.path.dirname(output_eval_file), exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = HybridTableLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        max_entity_candidate=args.max_entity_candidate,
        num_workers=0,
        mlm_probability=args.mlm_probability,
        ent_mlm_probability=args.ent_mlm_probability,
        is_train=False,
        sample_distribution=sample_distribution,
        use_cand=True,
        random_sample=False,
        use_visibility=False if args.no_visibility else True,
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    tok_eval_loss = 0.0
    ent_eval_loss = 0.0
    tok_eval_acc = 0.0
    ent_eval_acc = 0.0
    ent_eval_mr = 0.0
    core_ent_eval_map = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            _,
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_labels,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent_mask_type,
            input_ent,
            input_ent_type,
            input_ent_labels,
            input_ent_mask,
            candidate_entity_set,
            core_entity_set,
            _,
            _,
        ) = batch
        input_tok = input_tok.to(args.device)
        input_tok_type = input_tok_type.to(args.device)
        input_tok_pos = input_tok_pos.to(args.device)
        input_tok_mask = input_tok_mask.to(args.device)
        input_tok_labels = input_tok_labels.to(args.device)
        input_ent_text = input_ent_text.to(args.device)
        input_ent_text_length = input_ent_text_length.to(args.device)
        input_ent_mask_type = input_ent_mask_type.to(args.device)
        input_ent = input_ent.to(args.device)
        input_ent_type = input_ent_type.to(args.device)
        input_ent_mask = input_ent_mask.to(args.device)
        input_ent_labels = input_ent_labels.to(args.device)
        candidate_entity_set = candidate_entity_set.to(args.device)
        core_entity_set = core_entity_set.to(args.device)
        tok_outputs, ent_outputs = model(
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent_mask_type,
            input_ent,
            input_ent_type,
            input_ent_mask,
            candidate_entity_set,
            input_tok_labels,
            input_ent_labels,
        )
        tok_loss = tok_outputs[0]  # model outputs are always tuple in transformers (see doc)
        ent_loss = ent_outputs[0]
        tok_prediction_scores = tok_outputs[1]
        ent_prediction_scores = ent_outputs[1]
        tok_acc = accuracy(
            tok_prediction_scores.view(-1, config.vocab_size), input_tok_labels.view(-1), ignore_index=-1
        )
        ent_acc = accuracy(
            ent_prediction_scores.view(-1, config.max_entity_candidate), input_ent_labels.view(-1), ignore_index=-1
        )
        ent_mr = mean_rank(ent_prediction_scores.view(-1, config.max_entity_candidate), input_ent_labels.view(-1))
        core_ent_map = mean_average_precision(ent_prediction_scores[:, 1, :], core_entity_set)
        loss = tok_loss + ent_loss
        eval_loss += loss.mean().item()
        tok_eval_loss += tok_loss.mean().item()
        ent_eval_loss += ent_loss.mean().item()
        tok_eval_acc += tok_acc.item()
        ent_eval_acc += ent_acc.item()
        ent_eval_mr += ent_mr.item()
        core_ent_eval_map += core_ent_map.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    tok_eval_loss = tok_eval_loss / nb_eval_steps
    ent_eval_loss = ent_eval_loss / nb_eval_steps
    tok_eval_acc = tok_eval_acc / nb_eval_steps
    ent_eval_acc = ent_eval_acc / nb_eval_steps
    ent_eval_mr = ent_eval_mr / nb_eval_steps
    core_ent_eval_map = core_ent_eval_map / nb_eval_steps

    result = {
        "tok_eval_loss": tok_eval_loss,
        "tok_eval_acc": tok_eval_acc,
        "ent_eval_loss": ent_eval_loss,
        "ent_eval_acc": ent_eval_acc,
        "ent_eval_mr": ent_eval_mr,
        "core_ent_eval_map": core_ent_eval_map,
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w+") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


@torch.no_grad()
def evaluate_analysis(
    args,
    config,
    eval_dataset,
    model,
    output_file,
    prefix: str = "",
    sample_distribution: np.ndarray | None = None,
    log_dir: str | None = None,
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir if log_dir is None else log_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in {-1, 0}:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = EntityTableLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        max_entity_candidate=args.max_entity_candidate,
        num_workers=0,
        mlm_probability=args.mlm_probability,
        ent_mlm_probability=args.ent_mlm_probability,
        is_train=False,
        sample_distribution=sample_distribution,
        use_cand=args.use_cand,
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    ent_eval_acc = 0.0
    ent_eval_acc_5 = 0.0
    ent_eval_acc_10 = 0.0
    core_ent_eval_map = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            _,
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_labels,
            input_tok_mask,
            input_ent,
            input_ent_type,
            input_ent_labels,
            input_ent_mask,
            candidate_entity_set,
            core_entity_set,
            _,
            _,
        ) = batch
        input_tok = input_tok.to(args.device)
        input_tok_type = input_tok_type.to(args.device)
        input_tok_pos = input_tok_pos.to(args.device)
        input_tok_mask = input_tok_mask.to(args.device)
        input_tok_labels = input_tok_labels.to(args.device)
        input_ent = input_ent.to(args.device)
        input_ent_type = input_ent_type.to(args.device)
        input_ent_mask = input_ent_mask.to(args.device)
        input_ent_labels = input_ent_labels.to(args.device)
        candidate_entity_set = candidate_entity_set.to(args.device)
        core_entity_set = core_entity_set.to(args.device)
        tok_outputs, ent_outputs = model(
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent,
            input_ent_type,
            input_ent_mask,
            candidate_entity_set,
            input_tok_labels,
            input_ent_labels,
        )
        tok_outputs[1]
        ent_prediction_scores = ent_outputs[1]
        core_ent_map = mean_average_precision(ent_prediction_scores[:, 1, :], core_entity_set)
        core_ent_eval_map += core_ent_map.item()
        ent_acc = accuracy(
            ent_prediction_scores.view(-1, config.max_entity_candidate), input_ent_labels.view(-1), ignore_index=-1
        )
        ent_acc_5 = top_k_acc(
            ent_prediction_scores.view(-1, config.max_entity_candidate),
            input_ent_labels.view(-1),
            ignore_index=-1,
            k=5,
        )
        ent_acc_10 = top_k_acc(
            ent_prediction_scores.view(-1, config.max_entity_candidate),
            input_ent_labels.view(-1),
            ignore_index=-1,
            k=10,
        )
        ent_eval_acc += ent_acc.item()
        ent_eval_acc_5 += ent_acc_5.item()
        ent_eval_acc_10 += ent_acc_10.item()
        ent_sorted_predictions = torch.argsort(ent_prediction_scores, dim=-1, descending=True)
        for i in range(input_tok.shape[0]):
            tmp_str = []
            meta = ""
            headers = []
            for j in range(input_tok.shape[1]):
                if input_tok[i, j] == 0:
                    break
                if input_tok_pos[i, j] == 0:
                    if len(tmp_str) != 0:
                        if input_tok_type[i, j - 1] == 0:
                            meta = eval_dataset.tokenizer.decode(tmp_str)
                        elif input_tok_type[i, j - 1] == 1:
                            headers.append(eval_dataset.tokenizer.decode(tmp_str))
                    tmp_str = []
                if input_tok_labels[i, j] == -1:
                    tmp_str.append(input_tok[i, j].item())
                else:
                    tmp_str.append(input_tok_labels[i, j].item())
            if len(tmp_str) != 0:
                if input_tok_type[i, j - 1] == 0:
                    meta = eval_dataset.tokenizer.decode(tmp_str)
                elif input_tok_type[i, j - 1] == 1:
                    headers.append(eval_dataset.tokenizer.decode(tmp_str))
            output_file.write(meta + "\n")
            output_file.write("\t".join(headers) + "\n")
            for j in range(input_ent.shape[1]):
                if j == 0:
                    pgEnt = eval_dataset.entity_vocab[input_ent[i, j].item()]
                    output_file.write("pgEnt:\t%s\t%d" % (pgEnt["wiki_title"], pgEnt["count"]) + "\n")
                elif j == 1:
                    core_entities = [
                        candidate_entity_set[i, m] for m in range(args.max_entity_candidate) if core_entity_set[i, m]
                    ]
                    core_entities = [eval_dataset.entity_vocab[z.item()] for z in core_entities]
                    output_file.write("core entities:\n")
                    output_file.write(
                        "\t".join(["%s:%d" % (z["wiki_title"], z["count"]) for z in core_entities]) + "\n"
                    )
                    output_file.write("core entity predictions (top100):\n")
                    pred_core_entities = ent_sorted_predictions[i, 1, :100].tolist()
                    for z in pred_core_entities:
                        if core_entity_set[i, z]:
                            output_file.write(
                                "[%s:%f:%d]\t"
                                % (
                                    eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["wiki_title"],
                                    ent_prediction_scores[i, 1, z].item(),
                                    eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["count"],
                                )
                            )
                        else:
                            output_file.write(
                                "%s:%f:%d\t"
                                % (
                                    eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["wiki_title"],
                                    ent_prediction_scores[i, 1, z].item(),
                                    eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["count"],
                                )
                            )
                    output_file.write("\n")
                else:
                    ent = input_ent[i, j]
                    if ent == 0:
                        break
                    ent_label = input_ent_labels[i, j]
                    ent_type = input_ent_type[i, j]
                    pred_entities = ent_sorted_predictions[i, j, :100].tolist()
                    if ent_label == -1:
                        output_file.write(
                            "%s\t-1\t%d\n" % (eval_dataset.entity_vocab[ent.item()]["wiki_title"], ent_type)
                        )
                    else:
                        output_file.write(
                            "%s\t%s:%d\t%d\t"
                            % (
                                eval_dataset.entity_vocab[ent.item()]["wiki_title"],
                                eval_dataset.entity_vocab[candidate_entity_set[i, ent_label].item()]["wiki_title"],
                                eval_dataset.entity_vocab[candidate_entity_set[i, ent_label].item()]["count"],
                                ent_type,
                            )
                        )
                        for z in pred_entities:
                            if z == ent_label:
                                output_file.write(
                                    "[%s:%f:%d]"
                                    % (
                                        eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["wiki_title"],
                                        ent_prediction_scores[i, j, z],
                                        eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["count"],
                                    )
                                )
                                break
                            else:
                                output_file.write(
                                    "%s:%f:%d\t"
                                    % (
                                        eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["wiki_title"],
                                        ent_prediction_scores[i, j, z],
                                        eval_dataset.entity_vocab[candidate_entity_set[i, z].item()]["count"],
                                    )
                                )
                        output_file.write("\n")
            output_file.write("-" * 100 + "\n")
        nb_eval_steps += 1
    core_ent_eval_map = core_ent_eval_map / nb_eval_steps
    ent_eval_acc = ent_eval_acc / nb_eval_steps
    ent_eval_acc_5 = ent_eval_acc_5 / nb_eval_steps
    ent_eval_acc_10 = ent_eval_acc_10 / nb_eval_steps
    logger.info("core_ent_eval_map = %f" % core_ent_eval_map)
    logger.info("ent_eval_acc = %f" % ent_eval_acc)
    logger.info("ent_eval_acc_5 = %f" % ent_eval_acc_5)
    logger.info("ent_eval_acc_10 = %f" % ent_eval_acc_10)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data directory.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument("--resume", default="", type=str, help="The model checkpoint for continue training.")

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--ent_mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of entities to mask for masked language modeling loss",
    )
    parser.add_argument("--mall_probability", type=float, default=0.5, help="Ratio of mask both entity and text")
    parser.add_argument(
        "--max_entity_candidate", type=int, default=1000, help="num of entity candidate used in training"
    )
    parser.add_argument("--sample_distribution", action="store_true", help="generate candidate from distribution.")
    parser.add_argument("--use_cand", action="store_true", help="Train with collected candidates.")
    parser.add_argument("--random_sample", action="store_true", help="random sample candidates.")
    parser.add_argument("--no_visibility", action="store_true", help="no visibility matrix.")
    parser.add_argument("--exclusive_ent", type=int, default=0, help="whether to mask ent in the same column")

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_analysis", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--linear_scale_lr", action="store_true", help="Whether to linearly scale learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Linear warmup over warmup_epochs.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--dry_run", action="store_true", help="Sanity checks for training.")
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow the use of TF32 on NVIDIA Ampere GPUs")

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        args.n_gpu = dist.get_world_size()
    args.device = device

    # Linear scale the learning rate
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    if args.linear_scale_lr:
        args.scaled_learning_rate = (
            1e-4 * float(args.per_gpu_train_batch_size * world_size * args.gradient_accumulation_steps) / 50.0
        )
    else:
        args.scaled_learning_rate = args.learning_rate

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in {-1, 0} else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    seed_everything(args.seed)

    # Get model configuration
    config = TableConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.__dict__["max_entity_candidate"] = args.max_entity_candidate

    # Get date and time in format YYYY-MM-DD_HH-MM-SS
    dt_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Setup TensorboardLogger
    tb_logger = TensorBoardLogger(os.path.join(args.output_dir, "logs", "turl", "pre-train"), name=dt_now)
    tb_logger.experiment.add_text(
        "CLI arguments",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    tb_logger.experiment.add_text(
        "HF config",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.to_dict().items()])),
    )

    if args.local_rank in {-1, 0}:
        logger.info(
            "Training/evaluation parameters\n%s" % ("\n".join([f"{key}: {value}" for key, value in vars(args).items()]))
        )

    # Load pretrained model and tokenizer
    if args.local_rank not in {-1, 0}:
        dist.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Load entity vocab
    entity_vocab = load_entity_vocab(args.data_dir, ignore_bad_title=True, min_ent_count=2)
    if args.sample_distribution:
        sample_distribution = generate_vocab_distribution(entity_vocab)
    else:
        sample_distribution = None
    entity_wikid2id = {entity_vocab[x]["wiki_id"]: x for x in entity_vocab}

    assert config.ent_vocab_size == len(entity_vocab)

    # Create model
    model = HybridTableMaskedLM(config, is_simple=True)

    if args.do_train:
        if args.resume == "":
            lm_model_dir = "tiny-bert"
            lm_checkpoint = torch.load(lm_model_dir + "/pytorch_model.bin")
            model.load_pretrained(lm_checkpoint)
            new_ent_embeddings_dir = Path("./ent_embeddings")
            new_ent_embeddings_dir.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(new_ent_embeddings_dir / "new_ent_embeddings.bin"):
                origin_ent_embeddings = model.table.embeddings.ent_embeddings.weight.data.numpy()
                new_ent_embeddings = create_ent_embedding(args.data_dir, entity_wikid2id, origin_ent_embeddings)
                torch.save(new_ent_embeddings, str(new_ent_embeddings_dir / "new_ent_embeddings.bin"))
            else:
                new_ent_embeddings = torch.load(new_ent_embeddings_dir / "new_ent_embeddings.bin", map_location="cpu")
            model.table.embeddings.ent_embeddings.weight.data = torch.FloatTensor(new_ent_embeddings)
        else:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        model.to(args.device)

    if args.local_rank == 0:
        dist.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Training
    if args.do_train:
        if args.local_rank not in {-1, 0}:
            # Barrier to make sure only the first process in distributed training process the dataset
            # while the others will use the cache
            dist.barrier()

        train_dataset = WikiHybridTableDataset(
            args.data_dir,
            entity_vocab,
            max_cell=100,
            max_input_tok=350,
            max_input_ent=150,
            src="train",
            max_length=[50, 10, 10],
            force_new=False,
            tokenizer=None,
            dry_run=args.dry_run,
        )
        eval_dataset = WikiHybridTableDataset(
            args.data_dir,
            entity_vocab,
            max_cell=100,
            max_input_tok=350,
            max_input_ent=150,
            src="dev",
            max_length=[50, 10, 10],
            force_new=False,
            tokenizer=None,
            dry_run=args.dry_run,
        )

        assert config.vocab_size == len(train_dataset.tokenizer) and config.ent_vocab_size == len(
            train_dataset.entity_wikid2id
        ), "vocab size mismatch, vocab_size=%d, ent_vocab_size=%d" % (
            len(train_dataset.tokenizer),
            len(train_dataset.entity_wikid2id),
        )

        if args.local_rank == 0:
            dist.barrier()

        # Run the training loop
        global_step, tr_loss = train(
            args,
            config,
            train_dataset,
            model,
            tb_logger,
            eval_dataset=eval_dataset,
            sample_distribution=sample_distribution,
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and args.local_rank in {-1, 0}:
        output_dir = os.path.join(tb_logger.log_dir, "checkpoints", "checkpoint-last")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving last model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = getattr(model, "module", model)
        model_to_save.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in {-1, 0}:
        checkpoints_folder = os.path.join(tb_logger.log_dir, "checkpoints")
        checkpoints = [checkpoints_folder]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(checkpoints_folder + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", ", ".join(checkpoints))
        model = getattr(model, "module", model)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, WEIGHTS_NAME)))
            model.to(args.device)
            result = evaluate(args, config, eval_dataset, model, prefix=prefix, log_dir=tb_logger.log_dir)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_analysis and args.local_rank in {-1, 0}:
        import pdb

        pdb.set_trace()
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(tb_logger.log_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        checkpoint = checkpoints[-1]
        checkpoint = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        output_file = open(os.path.join(tb_logger.log_dir, "dev_analysis.txt"), "w", encoding="utf-8")
        model.load_state_dict(checkpoint)
        model.to(args.device)
        eval_dataset = WikiEntityTableDataset(
            args.data_dir,
            entity_vocab,
            max_cell=100,
            max_input_tok=350,
            max_input_ent=150,
            src="dev",
            max_length=[50, 10, 10],
            force_new=False,
            tokenizer=None,
        )
        evaluate_analysis(
            args,
            config,
            eval_dataset,
            getattr(model, "module", model),
            output_file,
            sample_distribution=sample_distribution,
            log_dir=tb_logger.log_dir,
        )
        output_file.flush()
        output_file.close()

    return results


if __name__ == "__main__":
    main()
