#!/usr/bin/env python3
"""Evaluate validation loss for the Accelerate MM-Mix direct-processor path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from odb_mm_mix import DirectReadMMMixDataset
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor

from mm_utils import make_model_collator
from train_accelerate import (
    configure_processor_pixels,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.getenv("ODB_ACCEL_EVAL_CHECKPOINT"))
    parser.add_argument(
        "--data",
        default=os.getenv("ODB_MM_MIX_DATA", "data/mm-mix-tmdb"),
    )
    parser.add_argument("--output-dir", default=os.getenv("ODB_ACCEL_EVAL_SAVE_DIR"))
    parser.add_argument(
        "--val-start",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_VAL_START", "196200")),
    )
    parser.add_argument(
        "--split-mode",
        choices=["tail", "lf_val_size"],
        default=os.getenv("ODB_ACCEL_EVAL_SPLIT_MODE", "lf_val_size"),
        help="Validation split. `lf_val_size` matches LLaMA-Factory TMDB val_size splitting.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=float(os.getenv("ODB_ACCEL_EVAL_VAL_SIZE", "0.05")),
        help="Validation size used when --split-mode=lf_val_size.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(os.getenv("ODB_ACCEL_EVAL_SPLIT_SEED", "42")),
        help="Seed used when --split-mode=lf_val_size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("ODB_ACCEL_EVAL_MAX_SAMPLES", "0")),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "16384")),
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_IMAGE_MAX_PIXELS", "9437184")),
    )
    parser.add_argument(
        "--processor-backend",
        choices=[
            "auto",
            "qwen_vl",
            "qwen3_vl",
            "llamafactory_qwen_vl",
            "generic",
            "hf",
            "processor",
        ],
        default=os.getenv("ODB_MM_MIX_PROCESSOR_BACKEND", "qwen_vl"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=int(os.getenv("ODB_MM_MIX_NUM_WORKERS", "4"))
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_PREFETCH_FACTOR", "16")),
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ODB_ACCEL_EVAL_BF16", "1").lower() in {"1", "true", "yes"},
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--limit-log-steps",
        type=int,
        default=int(os.getenv("ODB_ACCEL_EVAL_LOG_STEPS", "50")),
    )
    return parser.parse_args()


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def summarize_counts(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "min": 0,
            "p10": 0,
            "p50": 0,
            "p90": 0,
            "max": 0,
            "mean": 0.0,
        }
    ordered = sorted(values)

    def quantile(q: float) -> int:
        index = round((len(ordered) - 1) * q)
        return int(ordered[index])

    return {
        "min": int(ordered[0]),
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "max": int(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def build_eval_indices(
    args: argparse.Namespace, dataset_len: int
) -> tuple[list[int], dict[str, Any]]:
    if args.split_mode == "tail":
        if args.val_start < 0 or args.val_start >= dataset_len:
            raise SystemExit(
                f"val_start={args.val_start} is outside dataset length {dataset_len}"
            )
        end = dataset_len
        if args.max_samples > 0:
            end = min(end, args.val_start + args.max_samples)
        indices = list(range(args.val_start, end))
        return indices, {
            "split_mode": "tail",
            "val_start": args.val_start,
            "val_size": None,
            "split_seed": None,
        }

    if args.val_size <= 0:
        raise SystemExit("--val-size must be positive for --split-mode=lf_val_size")

    import numpy as np

    val_size = (
        int(args.val_size) if args.val_size > 1 else int(dataset_len * args.val_size)
    )
    val_size = max(1, min(val_size, dataset_len - 1))
    rng = np.random.default_rng(args.split_seed)
    indices = rng.permutation(dataset_len)[:val_size].tolist()
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    return [int(index) for index in indices], {
        "split_mode": "lf_val_size",
        "val_start": None,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "eval_indices_preview": [int(index) for index in indices[:10]],
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint or ODB_ACCEL_EVAL_CHECKPOINT is required")
    if not args.output_dir:
        raise SystemExit("--output-dir or ODB_ACCEL_EVAL_SAVE_DIR is required")

    checkpoint = Path(args.checkpoint)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    configure_processor_pixels(processor, image_max_pixels=args.image_max_pixels)
    raw_dataset = DirectReadMMMixDataset(
        data_path,
        processor=processor,
        max_length=args.max_length,
        image_max_pixels=args.image_max_pixels if args.image_max_pixels > 0 else None,
        processor_backend=args.processor_backend,
    )
    eval_indices, split_info = build_eval_indices(args, len(raw_dataset))
    dataset = Subset(raw_dataset, eval_indices)
    collator = make_model_collator(processor)
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(
        str(checkpoint),
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        attn_implementation=None,
    )
    model.to(device)
    model.eval()

    losses: list[float] = []
    total_loss = 0.0
    total_samples = 0
    total_label_tokens = 0
    token_weighted_loss = 0.0
    label_tokens_per_sample: list[int] = []
    use_amp = device.type == "cuda" and args.bf16
    progress = tqdm(dataloader, desc="eval", dynamic_ncols=True)
    with torch.no_grad():
        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)
            labels = batch.get("labels")
            batch_label_tokens = 0
            if isinstance(labels, torch.Tensor):
                per_sample = (labels != -100).view(labels.shape[0], -1).sum(dim=1)
                counts = [int(value.item()) for value in per_sample]
                label_tokens_per_sample.extend(counts)
                batch_label_tokens = sum(counts)
                total_label_tokens += batch_label_tokens
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                output = model(**batch)
            loss = float(output.loss.detach().float().item())
            batch_size = int(batch["input_ids"].shape[0])
            losses.append(loss)
            total_loss += loss * batch_size
            token_weighted_loss += loss * batch_label_tokens
            total_samples += batch_size
            if args.limit_log_steps > 0 and step % args.limit_log_steps == 0:
                progress.set_postfix(eval_loss=total_loss / max(total_samples, 1))

    eval_loss = total_loss / max(total_samples, 1)
    token_weighted_eval_loss = token_weighted_loss / max(total_label_tokens, 1)
    result = {
        "checkpoint": str(checkpoint),
        "data": str(data_path),
        **split_info,
        "eval_samples": total_samples,
        "eval_loss": eval_loss,
        "mean_batch_loss": sum(losses) / max(len(losses), 1),
        "loss_aggregation": "sample_mean_of_model_loss",
        "token_weighted_eval_loss": token_weighted_eval_loss,
        "label_tokens": total_label_tokens,
        "label_tokens_per_sample": summarize_counts(label_tokens_per_sample),
        "max_length": args.max_length,
        "image_max_pixels": args.image_max_pixels,
        "processor_backend": args.processor_backend,
    }
    for name in ("eval_results.json", "all_results.json"):
        (output_dir / name).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("[hf-valloss] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
