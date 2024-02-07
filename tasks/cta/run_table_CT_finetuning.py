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
from typing import Type

import torch
import torch.distributed as dist
from lightning import seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

from src.data_loader.ct_wiki_data_loaders import WikiCTDataset
from src.model.configuration import TableConfig
from src.model.metric import average_precision
from src.model.model import HybridTableCT
from src.utils.util import load_entity_vocab, load_type_vocab, rotate_checkpoints

logger = logging.getLogger(__name__)


def train(
    args: argparse.Namespace,
    config: TableConfig,
    train_dataset: Dataset,
    dataloader_cls: Type[DataLoader],
    model: HybridTableCT,
    tb_logger: TensorBoardLogger,
    eval_dataset: Dataset | None = None,
):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = dataloader_cls(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, is_train=True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.table.named_parameters() if (not any(nd in n for nd in no_decay))],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.table.named_parameters() if (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.cls.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * 10 if args.cls_learning_rate == 0 else args.cls_learning_rate,
        },
        {
            "params": [p for n, p in model.cls.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 10 if args.cls_learning_rate == 0 else args.cls_learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

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

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_map, logging_map = 0.0, 0.0

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
                table_id,
                input_tok,
                input_tok_type,
                input_tok_pos,
                input_tok_mask,
                input_ent_text,
                input_ent_text_length,
                input_ent,
                input_ent_type,
                input_ent_mask,
                column_entity_mask,
                column_header_mask,
                labels_mask,
                labels,
            ) = batch
            input_tok = input_tok.to(args.device)
            input_tok_type = input_tok_type.to(args.device)
            input_tok_pos = input_tok_pos.to(args.device)
            input_tok_mask = input_tok_mask.to(args.device)
            input_ent_text = input_ent_text.to(args.device)
            input_ent_text_length = input_ent_text_length.to(args.device)
            input_ent = input_ent.to(args.device)
            input_ent_type = input_ent_type.to(args.device)
            input_ent_mask = input_ent_mask.to(args.device)
            column_entity_mask = column_entity_mask.to(args.device)
            column_header_mask = column_header_mask.to(args.device)
            labels_mask = labels_mask.to(args.device)
            labels = labels.to(args.device)
            if args.mode == 1:
                input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
                input_tok = None
                input_tok_type = None
                input_tok_pos = None
                input_tok_mask = None
            elif args.mode == 2:
                input_tok_mask = input_tok_mask[:, :, : input_tok_mask.shape[1]]
                input_ent_text = None
                input_ent_text_length = None
                input_ent = None
                input_ent_type = None
                input_ent_mask = None
            elif args.mode == 3:
                input_ent = None
            elif args.mode == 4:
                input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
                input_tok = None
                input_tok_type = None
                input_tok_pos = None
                input_tok_mask = None
                input_ent = None
            elif args.mode == 5:
                input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
                input_tok = None
                input_tok_type = None
                input_tok_pos = None
                input_tok_mask = None
                input_ent_text = None
                input_ent_text_length = None

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
                    outputs = model(
                        input_tok,
                        input_tok_type,
                        input_tok_pos,
                        input_tok_mask,
                        input_ent_text,
                        input_ent_text_length,
                        input_ent,
                        input_ent_type,
                        input_ent_mask,
                        column_entity_mask,
                        column_header_mask,
                        labels_mask,
                        labels,
                    )
                    # model outputs are always tuple in transformers (see doc)
                    loss = outputs[0]

                prediction_scores = outputs[1]
                ap = average_precision(
                    prediction_scores.view(-1, config.class_num), labels.view((-1, config.class_num))
                )
                map = (ap * labels_mask.view(-1)).sum() / labels_mask.sum()

                # Loss reduction
                loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Scaled loss backward for mixed precision training
                scaler.scale(loss).backward()

            tr_loss += loss.item()
            tr_map += map.item()

            if not is_accumulating:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
                        logger.info("***** Train results *****")
                        logger.info("  loss = %s", str((tr_loss - logging_loss) / args.logging_steps))
                        logger.info(
                            "  map = %s",
                            str((tr_map - logging_map) / (args.gradient_accumulation_steps * args.logging_steps)),
                        )
                        results = evaluate(
                            args,
                            config,
                            eval_dataset,
                            dataloader_cls,
                            getattr(model, "module", model),
                            prefix="checkpoints/checkpoint-{}".format(global_step),
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
                            "map": (tr_map - logging_map) / (args.gradient_accumulation_steps * args.logging_steps),
                        },
                        global_step,
                    )
                    logging_map = tr_map
                    logging_loss = tr_loss

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

    return global_step, tr_loss / global_step


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    config: TableConfig,
    eval_dataset: Dataset,
    dataloader_cls: Type[DataLoader],
    model: HybridTableCT,
    prefix: str = "",
    log_dir: str | None = None,
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir if log_dir is None else log_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in {-1, 0}:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = dataloader_cls(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, is_train=False
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_map = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            table_id,
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent,
            input_ent_type,
            input_ent_mask,
            column_entity_mask,
            column_header_mask,
            labels_mask,
            labels,
        ) = batch
        input_tok = input_tok.to(args.device)
        input_tok_type = input_tok_type.to(args.device)
        input_tok_pos = input_tok_pos.to(args.device)
        input_tok_mask = input_tok_mask.to(args.device)
        input_ent_text = input_ent_text.to(args.device)
        input_ent_text_length = input_ent_text_length.to(args.device)
        input_ent = input_ent.to(args.device)
        input_ent_type = input_ent_type.to(args.device)
        input_ent_mask = input_ent_mask.to(args.device)
        column_entity_mask = column_entity_mask.to(args.device)
        column_header_mask = column_header_mask.to(args.device)
        labels_mask = labels_mask.to(args.device)
        labels = labels.to(args.device)
        if args.mode == 1:
            input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
            input_tok = None
            input_tok_type = None
            input_tok_pos = None
            input_tok_mask = None
        elif args.mode == 2:
            input_tok_mask = input_tok_mask[:, :, : input_tok_mask.shape[1]]
            input_ent_text = None
            input_ent_text_length = None
            input_ent = None
            input_ent_type = None
            input_ent_mask = None
        elif args.mode == 3:
            input_ent = None
        elif args.mode == 4:
            input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
            input_tok = None
            input_tok_type = None
            input_tok_pos = None
            input_tok_mask = None
            input_ent = None
        elif args.mode == 5:
            input_ent_mask = input_ent_mask[:, :, input_tok_mask.shape[1] :]
            input_tok = None
            input_tok_type = None
            input_tok_pos = None
            input_tok_mask = None
            input_ent_text = None
            input_ent_text_length = None
        outputs = model(
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent,
            input_ent_type,
            input_ent_mask,
            column_entity_mask,
            column_header_mask,
            labels_mask,
            labels,
        )
        loss = outputs[0]
        prediction_scores = outputs[1]
        ap = average_precision(prediction_scores.view(-1, config.class_num), labels.view((-1, config.class_num)))
        map = (ap * labels_mask.view(-1)).sum() / labels_mask.sum()
        eval_loss += loss.mean().item()
        eval_map += map.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_map = eval_map / nb_eval_steps

    result = {"eval_loss": eval_loss, "eval_map": eval_map}
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result


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
    parser.add_argument("--mode", default=0, type=int, help="0: use both;1: use table;2: use entity")
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
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--cls_learning_rate", default=0, type=float, help="The initial learning rate for cls layer.")
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
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
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
    parser.add_argument("--base_lr", type=float, default=5e-5, help="The base learning rate used to train the model.")
    parser.add_argument(
        "--base_effective_batch_size",
        type=float,
        default=40,
        help="The base effective batch size used to train the model. "
        "The effective batch size is computed as "
        "base_effective_batch_size = per_gpu_train_batch_size * gradient_accumulation_steps * n_gpu.",
    )
    parser.add_argument(
        "--loader_type",
        default="wikitables",
        const="wikitables",
        nargs="?",
        choices=["wikitables", "semtab"],
    )

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

    # Load pretrained model and tokenizer
    if args.local_rank not in {-1, 0}:
        dist.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Get model config
    config = TableConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Get date and time in format YYYY-MM-DD_HH-MM-SS
    dt_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Setup TensorboardLogger
    tb_logger = TensorBoardLogger(os.path.join(args.output_dir, "logs", "turl", "fine-tuning-ct"), name=dt_now)
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

    type_vocab = load_type_vocab(args.data_dir)
    config.class_num = len(type_vocab)
    config.mode = args.mode

    # Load pre-trained model
    model = HybridTableCT(config, is_simple=True)
    if args.do_train:
        lm_checkpoint = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"))
        model.load_pretrained(lm_checkpoint)
        model.to(args.device)

    # Get dataloader class
    if args.loader_type == "wikitables":
        from src.data_loader.ct_wiki_data_loaders import CTLoader

        dataloader_cls = CTLoader
    elif args.loader_type == "semtab":
        from src.data_loader.ct_semcol_data_loaders import CTLoader

        dataloader_cls = CTLoader

    if args.local_rank == 0:
        dist.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Training
    if args.do_train:
        if args.local_rank not in {-1, 0}:
            dist.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        entity_vocab = load_entity_vocab(args.data_dir, ignore_bad_title=True, min_ent_count=2)
        train_dataset = WikiCTDataset(
            args.data_dir,
            entity_vocab,
            type_vocab,
            max_input_tok=500,
            src="train",
            max_length=[50, 10, 10],
            force_new=False,
            tokenizer=None,
        )
        eval_dataset = WikiCTDataset(
            args.data_dir,
            entity_vocab,
            type_vocab,
            max_input_tok=500,
            src="dev",
            max_length=[50, 10, 10],
            force_new=False,
            tokenizer=None,
        )
        assert config.vocab_size == len(train_dataset.tokenizer), "vocab size mismatch, vocab_size=%d" % (
            len(train_dataset.tokenizer)
        )

        if args.local_rank == 0:
            dist.barrier()

        global_step, tr_loss = train(
            args,
            config,
            train_dataset,
            dataloader_cls,
            model,
            tb_logger,
            eval_dataset=eval_dataset,
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and args.local_rank in {-1, 0}:
        output_dir = os.path.join(tb_logger.log_dir, "checkpoints", "checkpoint-last")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = getattr(model, "module", model)
        model_to_save.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in {-1, 0}:
        checkpoints_folder = os.path.join(tb_logger.log_dir, "checkpoints")
        checkpoints = [checkpoints_folder]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, WEIGHTS_NAME)))
            model.to(args.device)
            result = evaluate(
                args,
                config,
                eval_dataset,
                dataloader_cls,
                model,
                prefix=prefix,
                log_dir=tb_logger.log_dir,
            )
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
