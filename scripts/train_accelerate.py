#!/usr/bin/env python3
"""Accelerate real-processor MM-Mix example using the ODB pip package."""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from pathlib import Path
from typing import Any

import odb
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from mm_utils import add_qwen_vl_position_ids, make_model_collator, reset_qwen_vl_rope_cache
from odb.integrations.accelerate import configure_accelerator
from odb_mm_mix import DirectReadMMMixDataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor


VISION_MODEL_KEYS = ("visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list")
MULTIMODAL_PROJECTOR_KEYS = ("visual.merger",)
LANGUAGE_MODEL_KEYS = ("language_model", "lm_head")


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def shutdown_odb_iterator(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def cleanup_after_training(accelerator: Accelerator, train_loader: Any, model: Any, optimizer: Any) -> None:
    """Release distributed resources after all training/eval work has finished."""
    active_iterator = getattr(train_loader, "_odb_active_iterator", None)
    if active_iterator is not None:
        with contextlib.suppress(Exception):
            shutdown_odb_iterator(active_iterator)
        with contextlib.suppress(Exception):
            setattr(train_loader, "_odb_active_iterator", None)

    with contextlib.suppress(Exception):
        accelerator.free_memory(model, optimizer)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()

    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    for child in multiprocessing.active_children():
        with contextlib.suppress(Exception):
            child.terminate()
        with contextlib.suppress(Exception):
            child.join(timeout=2)


def exit_successfully_after_teardown() -> None:
    """Avoid framework background workers keeping a finished training job alive."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    os._exit(0)


def count_records(path: Path) -> int:
    metadata = path / "metadata.json"
    if metadata.exists():
        try:
            return int(json.loads(metadata.read_text(encoding="utf-8")).get("num_records") or 0)
        except Exception:
            pass
    records = path / "records.jsonl"
    if not records.exists():
        return 0
    with records.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def copy_tree_if_needed(source: Path, target: Path, *, force: bool = False) -> Path:
    if count_records(source) <= 0:
        raise SystemExit(f"source TMDB is missing or empty: {source}")
    if force and target.exists():
        shutil.rmtree(target)
    if count_records(target) == count_records(source):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        target.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.check_call([rsync, "-a", "--delete", f"{source}/", f"{target}/"])
    else:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return target


def load_model(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
    dtype: torch.dtype,
    attn_implementation: str | None,
):
    import transformers

    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    init_kwargs: dict[str, Any] = {}
    if attn_implementation:
        init_kwargs["attn_implementation"] = attn_implementation
    try:
        return model_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **init_kwargs,
        )
    except ValueError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **init_kwargs,
        )


def configure_processor_pixels(processor: Any, *, image_max_pixels: int | None) -> None:
    if image_max_pixels is None or image_max_pixels <= 0:
        return
    targets = [processor, getattr(processor, "image_processor", None)]
    for target in targets:
        if target is None:
            continue
        for name in ("max_pixels", "image_max_pixels"):
            if hasattr(target, name):
                try:
                    setattr(target, name, int(image_max_pixels))
                except Exception:
                    pass


def configure_trainable_parameters(
    model: torch.nn.Module,
    trainable_keywords: tuple[str, ...],
    *,
    freeze_vision_tower: bool,
    freeze_multimodal_projector: bool,
    freeze_language_model: bool,
) -> int:
    full_train = any(keyword.lower() in {"*", "all", "full"} for keyword in trainable_keywords)
    frozen_keys: list[str] = []
    if freeze_vision_tower:
        frozen_keys.extend(VISION_MODEL_KEYS)
    if freeze_multimodal_projector:
        frozen_keys.extend(MULTIMODAL_PROJECTOR_KEYS)
    if freeze_language_model:
        frozen_keys.extend(LANGUAGE_MODEL_KEYS)

    trainable = 0
    for name, param in model.named_parameters():
        keep = full_train or any(keyword in name for keyword in trainable_keywords)
        if any(key in name for key in frozen_keys):
            keep = False
        param.requires_grad_(keep)
        if keep:
            trainable += param.numel()
    return trainable


def configure_training_memory(model: torch.nn.Module, *, gradient_checkpointing: bool) -> None:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        try:
            config.use_cache = False
        except Exception:
            pass
    if not gradient_checkpointing:
        return
    if getattr(model, "supports_gradient_checkpointing", False):
        try:
            from torch.utils.checkpoint import checkpoint

            gradient_checkpointing_func = _custom_gradient_checkpointing_func(partial(checkpoint, use_reentrant=True))
            if "value" in inspect.signature(model._set_gradient_checkpointing).parameters:
                model.apply(partial(model._set_gradient_checkpointing, value=True))
            else:
                model._set_gradient_checkpointing(
                    enable=True,
                    gradient_checkpointing_func=gradient_checkpointing_func,
                )
        except Exception:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            except TypeError:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
            except Exception:
                pass
    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        try:
            enable_input_require_grads()
        except Exception:
            pass


def _custom_gradient_checkpointing_func(gradient_checkpointing_func):
    """LLaMA-Factory-style GC wrapper for framework-native examples."""

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func, *args, **kwargs):
        module = func.func.__self__ if isinstance(func, partial) else func.__self__
        has_grad = any(param.requires_grad for param in module.parameters())
        if not has_grad:
            return func(*args, **kwargs)
        for arg in args:
            if torch.is_tensor(arg) and torch.is_floating_point(arg):
                arg.requires_grad_(True)
                break
        return gradient_checkpointing_func(func, *args, **kwargs)

    return custom_gradient_checkpointing_func


def model_runtime_info(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    return {
        "gradient_checkpointing_active": bool(getattr(model, "is_gradient_checkpointing", False)),
        "attn_implementation": getattr(config, "_attn_implementation", None)
        or getattr(config, "attn_implementation", None),
        "text_attn_implementation": getattr(text_config, "_attn_implementation", None)
        or getattr(text_config, "attn_implementation", None),
        "vision_attn_implementation": getattr(vision_config, "_attn_implementation", None)
        or getattr(vision_config, "attn_implementation", None),
    }


def parse_args() -> argparse.Namespace:
    gradient_checkpointing_default = env_flag("ODB_MM_MIX_GRADIENT_CHECKPOINTING", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("ODB_MM_MIX_DATA", "data/mm-mix-tmdb"))
    parser.add_argument("--source-data", default=os.getenv("ODB_MM_MIX_SOURCE_DATA"))
    parser.add_argument("--local-data", default=os.getenv("ODB_MM_MIX_LOCAL_DATA"))
    parser.add_argument("--force-local-copy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default=os.getenv("ODB_MM_MIX_MODEL", "Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default=os.getenv("ODB_MM_MIX_ATTN_IMPLEMENTATION", ""),
        help="Optional HF attention implementation, e.g. sdpa or flash_attention_2.",
    )
    parser.add_argument("--loader", choices=["odb", "standard"], default=os.getenv("ODB_MM_MIX_LOADER", "odb"))
    parser.add_argument("--output-dir", default=os.getenv("ODB_MM_MIX_OUTPUT_DIR", "outputs/accelerate-real"))
    parser.add_argument("--token-budget", type=int, default=int(os.getenv("ODB_MM_MIX_TOKEN_BUDGET", "12288")))
    parser.add_argument("--buffer-size", type=int, default=int(os.getenv("ODB_MM_MIX_BUFFER_SIZE", "1024")))
    parser.add_argument("--max-patches", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_PATCHES", "0")))
    parser.add_argument("--fixed-batch-size", type=int, default=int(os.getenv("ODB_MM_MIX_FIXED_BATCH_SIZE", "1")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "16384")))
    parser.add_argument("--train-size", type=int, default=int(os.getenv("ODB_MM_MIX_TRAIN_SIZE", "0")))
    parser.add_argument(
        "--split-mode",
        choices=["prefix", "lf_val_size"],
        default=os.getenv("ODB_MM_MIX_SPLIT_MODE", "lf_val_size"),
        help="Training split. `lf_val_size` matches LLaMA-Factory TMDB val_size splitting and trains on the complement.",
    )
    parser.add_argument("--val-size", type=float, default=float(os.getenv("ODB_MM_MIX_VAL_SIZE", "0.05")))
    parser.add_argument("--split-seed", type=int, default=int(os.getenv("ODB_MM_MIX_SPLIT_SEED", "42")))
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_IMAGE_MAX_PIXELS", "9437184")),
        help="Downscale images above this pixel budget before Qwen-VL vision-token expansion; set 0 to disable.",
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
        default=os.getenv("ODB_MM_MIX_PROCESSOR_BACKEND", "auto"),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_STEPS", "0")))
    parser.add_argument("--num-train-epochs", type=float, default=float(os.getenv("ODB_MM_MIX_EPOCHS", "1.0")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("ODB_MM_MIX_NUM_WORKERS", "4")))
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_PREFETCH_FACTOR", "0")),
        help="Worker prefetch factor. Use 0 for the LF-aligned default: 2 for Standard and 512 for ODB.",
    )
    parser.add_argument("--lr", type=float, default=float(os.getenv("ODB_MM_MIX_LR", "1e-5")))
    parser.add_argument("--lr-scheduler-type", default=os.getenv("ODB_MM_MIX_LR_SCHEDULER_TYPE", "cosine"))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.getenv("ODB_MM_MIX_WARMUP_RATIO", "0.03")))
    parser.add_argument("--max-grad-norm", type=float, default=float(os.getenv("ODB_MM_MIX_MAX_GRAD_NORM", "4.0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("ODB_MM_MIX_SEED", "42")))
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=gradient_checkpointing_default,
    )
    parser.add_argument("--join", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-scaling", default=os.getenv("ODB_MM_MIX_LOSS_SCALING", "exact"))
    parser.add_argument("--deepspeed-config", default=os.getenv("ODB_MM_MIX_DEEPSPEED_CONFIG"))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ODB_MM_MIX_SAVE_FINAL_MODEL", False),
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--trainable-keywords",
        default=os.getenv("ODB_MM_MIX_TRAINABLE_KEYWORDS", "full"),
        help=(
            "Comma-separated parameter-name fragments to train. With the default freeze flags, "
            "'full' matches LLaMA-Factory VLM full-SFT semantics: train the language model while "
            "freezing the vision tower and multimodal projector."
        ),
    )
    parser.add_argument(
        "--freeze-vision-tower",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ODB_MM_MIX_FREEZE_VISION_TOWER", True),
    )
    parser.add_argument(
        "--freeze-multimodal-projector",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ODB_MM_MIX_FREEZE_MULTIMODAL_PROJECTOR", True),
    )
    parser.add_argument(
        "--freeze-language-model",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ODB_MM_MIX_FREEZE_LANGUAGE_MODEL", False),
    )
    return parser.parse_args()


def build_train_indices(args: argparse.Namespace, dataset_len: int) -> tuple[list[int], dict[str, Any]]:
    if args.split_mode == "prefix":
        indices = list(range(dataset_len)) if args.train_size <= 0 else list(range(args.train_size))
        if len(indices) > dataset_len:
            raise SystemExit(f"train_size={args.train_size} exceeds dataset size {dataset_len}")
        return indices, {
            "split_mode": "prefix",
            "train_size_arg": args.train_size,
            "val_size": None,
            "split_seed": None,
            "train_indices_preview": indices[:10],
            "eval_indices_preview": None,
        }

    if args.val_size <= 0:
        raise SystemExit("--val-size must be positive for --split-mode=lf_val_size")

    import numpy as np

    val_size = int(args.val_size) if args.val_size > 1 else int(dataset_len * args.val_size)
    val_size = max(1, min(val_size, dataset_len - 1))
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(dataset_len).tolist()
    eval_indices = [int(index) for index in perm[:val_size]]
    train_indices = [int(index) for index in perm[val_size:]]
    if args.train_size > 0:
        if args.train_size > len(train_indices):
            raise SystemExit(f"train_size={args.train_size} exceeds LF-split train size {len(train_indices)}")
        train_indices = train_indices[: args.train_size]
    return train_indices, {
        "split_mode": "lf_val_size",
        "train_size_arg": args.train_size,
        "val_size": args.val_size,
        "split_seed": args.split_seed,
        "train_indices_preview": train_indices[:10],
        "eval_indices_preview": eval_indices[:10],
    }


def effective_prefetch_factor(args: argparse.Namespace) -> int:
    if args.prefetch_factor > 0:
        return args.prefetch_factor
    return 512 if args.loader == "odb" else 2


def make_train_dataloader(args: argparse.Namespace, dataset, collator) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": 1 if args.loader == "odb" else args.fixed_batch_size,
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = effective_prefetch_factor(args)
    if args.loader == "standard":
        return DataLoader(dataset, **kwargs)
    return odb.ODBDataLoader(
        dataset,
        token_budget=args.token_budget,
        buffer_size=args.buffer_size,
        loss_scaling=args.loss_scaling,
        join=args.join,
        max_patches=args.max_patches,
        **kwargs,
    )


def lr_scale(progress: float, *, scheduler_type: str, warmup_ratio: float) -> float:
    progress = max(0.0, min(float(progress), 1.0))
    if warmup_ratio > 0 and progress < warmup_ratio:
        return max(progress / warmup_ratio, 1e-8)
    if scheduler_type == "cosine":
        denom = max(1e-8, 1.0 - max(warmup_ratio, 0.0))
        cosine_progress = (progress - max(warmup_ratio, 0.0)) / denom
        return 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(cosine_progress, 1.0))))
    return 1.0


def set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lr: float, scale: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def reduce_loss_for_logging(accelerator: Accelerator, loss: torch.Tensor) -> float:
    gathered = accelerator.gather_for_metrics(loss.detach().float().reshape(1))
    return float(gathered.mean().cpu().item())


def gather_sample_count(accelerator: Accelerator, local_samples: int, device: torch.device) -> int:
    value = torch.tensor([int(local_samples)], device=device, dtype=torch.long)
    gathered = accelerator.gather_for_metrics(value)
    return int(gathered.sum().cpu().item())


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def batch_debug_summary(batch: dict[str, Any], *, step: int, rank: int) -> str:
    input_ids = batch.get("input_ids")
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    position_ids = batch.get("position_ids")
    parts = [f"rank={rank}", f"step={step}"]
    if isinstance(input_ids, torch.Tensor):
        parts.append(f"input_shape={tuple(input_ids.shape)}")
    if isinstance(pixel_values, torch.Tensor):
        parts.append(f"pixel_shape={tuple(pixel_values.shape)}")
        parts.append(f"pixel_dtype={pixel_values.dtype}")
    if isinstance(image_grid_thw, torch.Tensor):
        parts.append(f"image_grid_shape={tuple(image_grid_thw.shape)}")
        parts.append(f"image_grid_sum={int(image_grid_thw.prod(dim=1).sum().item())}")
    if isinstance(position_ids, torch.Tensor):
        parts.append(f"position_shape={tuple(position_ids.shape)}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        parts.append(f"cuda_alloc_gb={torch.cuda.memory_allocated(device) / (1024**3):.2f}")
        parts.append(f"cuda_reserved_gb={torch.cuda.memory_reserved(device) / (1024**3):.2f}")
    return "[odb-mm-mix-batch] " + " ".join(parts)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)

    mixed_precision = "bf16" if args.bf16 else "fp16" if args.fp16 else "no"
    deepspeed_plugin = None
    if args.deepspeed_config:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=args.deepspeed_config,
            gradient_accumulation_steps=1,
            gradient_clipping=args.max_grad_norm if args.max_grad_norm > 0 else None,
        )
    accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin)

    data_path = Path(args.data)
    if args.source_data or args.local_data:
        data_path = copy_tree_if_needed(
            Path(args.source_data or args.data),
            Path(args.local_data or args.data),
            force=args.force_local_copy,
        )
    if count_records(data_path) <= 0:
        raise SystemExit(f"No records found in {data_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    configure_processor_pixels(processor, image_max_pixels=args.image_max_pixels)
    model = load_model(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        attn_implementation=args.attn_implementation or None,
    )
    configure_training_memory(model, gradient_checkpointing=args.gradient_checkpointing)
    trainable_keywords = tuple(x.strip() for x in args.trainable_keywords.split(",") if x.strip())
    trainable = configure_trainable_parameters(
        model,
        trainable_keywords,
        freeze_vision_tower=args.freeze_vision_tower,
        freeze_multimodal_projector=args.freeze_multimodal_projector,
        freeze_language_model=args.freeze_language_model,
    )
    if trainable <= 0:
        raise SystemExit(f"No trainable parameters matched: {trainable_keywords}")

    raw_dataset = DirectReadMMMixDataset(
        data_path,
        processor=processor,
        max_length=args.max_length,
        image_max_pixels=args.image_max_pixels if args.image_max_pixels > 0 else None,
        processor_backend=args.processor_backend,
    )
    train_indices, split_info = build_train_indices(args, len(raw_dataset))
    dataset = Subset(raw_dataset, train_indices)
    collator = make_model_collator(processor, compute_dtype=dtype)
    train_loader = make_train_dataloader(args, dataset, collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    handle = getattr(train_loader, "odb_handle", None)
    if args.loader == "odb":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    bridge = None
    if args.loader == "odb":
        bridge = configure_accelerator(
            accelerator,
            train_loader,
            handle=handle,
            sample_budget=len(dataset),
            loss_scaling=args.loss_scaling,
        )

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "loader": args.loader,
                    "data": str(data_path),
                    "raw_records": len(raw_dataset),
                    "records": len(dataset),
                    **split_info,
                    "model": args.model,
                    "attn_implementation_arg": args.attn_implementation or None,
                    "trainable_parameters": trainable,
                    "trainable_keywords": list(trainable_keywords),
                    "freeze_vision_tower": args.freeze_vision_tower,
                    "freeze_multimodal_projector": args.freeze_multimodal_projector,
                    "freeze_language_model": args.freeze_language_model,
                    "token_budget": args.token_budget if args.loader == "odb" else None,
                    "fixed_batch_size": args.fixed_batch_size if args.loader == "standard" else None,
                    "max_length": args.max_length,
                    "image_max_pixels": args.image_max_pixels,
                    "processor_backend": args.processor_backend,
                    "num_workers": args.num_workers,
                    "prefetch_factor": effective_prefetch_factor(args) if args.num_workers > 0 else None,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    **model_runtime_info(model),
                    "deepspeed_config": args.deepspeed_config,
                    "max_steps": args.max_steps,
                    "world_size": accelerator.num_processes,
                },
                indent=2,
            ),
            flush=True,
        )

    accelerator.wait_for_everyone()
    start = time.perf_counter()
    total_loss = 0.0
    steps = 0
    emitted_samples = 0
    sample_budget = len(dataset)
    epochs = max(1, int(math.ceil(args.num_train_epochs)))
    join_context = bridge.join_uneven_inputs([model]) if bridge is not None else None
    context = join_context if join_context is not None else torch.no_grad()

    if bridge is None:

        class _NoopContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        context = _NoopContext()

    with context:
        for _epoch in range(epochs):
            train_iterator = iter(train_loader)
            if bridge is not None:
                setattr(train_loader, "_odb_active_iterator", train_iterator)
            try:
                for batch in train_iterator:
                    if bridge is not None:
                        batch = move_batch_to_device(batch, accelerator.device)
                    info = bridge.consume_batch(batch) if bridge is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    reset_qwen_vl_rope_cache(model)
                    batch = add_qwen_vl_position_ids(batch, model)
                    if os.getenv("ODB_MM_MIX_LOG_BATCH_SHAPES", "").lower() in {"1", "true", "yes", "on"}:
                        print(
                            batch_debug_summary(
                                batch,
                                step=steps + 1,
                                rank=accelerator.process_index,
                            ),
                            flush=True,
                        )
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    reset_qwen_vl_rope_cache(model)
                    if bridge is None:
                        accelerator.backward(raw_loss)
                        local_samples = int(batch["input_ids"].shape[0])
                        emitted_samples += gather_sample_count(accelerator, local_samples, raw_loss.device)
                    else:
                        bridge.backward(raw_loss, info=info)
                        emitted_samples = int(bridge.state.emitted_samples)
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if bridge is not None:
                        bridge.mark_optimizer_step()
                    steps += 1
                    logged_loss = reduce_loss_for_logging(accelerator, raw_loss)
                    total_loss += logged_loss
                    progress = emitted_samples / sample_budget if sample_budget else 0.0
                    set_optimizer_lr(
                        optimizer,
                        args.lr,
                        lr_scale(
                            progress,
                            scheduler_type=args.lr_scheduler_type,
                            warmup_ratio=args.warmup_ratio,
                        ),
                    )
                    if accelerator.is_main_process and (steps == 1 or steps % args.logging_steps == 0):
                        accelerator.print(
                            f"step={steps} loss={total_loss / steps:.4f} emitted={emitted_samples} "
                            f"lr={optimizer.param_groups[0]['lr']:.3e}"
                        )
                    if args.max_steps > 0 and steps >= args.max_steps:
                        break
                    if emitted_samples >= sample_budget:
                        break
            finally:
                if bridge is not None:
                    shutdown_odb_iterator(train_iterator)
                    setattr(train_loader, "_odb_active_iterator", None)
            if args.max_steps > 0 and steps >= args.max_steps:
                break
            if emitted_samples >= sample_budget:
                break

    accelerator.wait_for_everyone()
    runtime = time.perf_counter() - start
    if accelerator.is_main_process:
        summary = {
            "loader": args.loader,
            "global_step": steps,
            "emitted_samples": int(emitted_samples),
            "mean_emitted_samples_per_step": emitted_samples / steps if steps else None,
            "effective_emitted_samples_per_second": emitted_samples / runtime if runtime > 0 else None,
            "runtime_seconds": runtime,
            "train_loss": total_loss / max(steps, 1),
            "world_size": accelerator.num_processes,
            "split": split_info,
            "token_budget": args.token_budget if args.loader == "odb" else None,
            "buffer_size": args.buffer_size if args.loader == "odb" else None,
            "loss_scaling": args.loss_scaling if args.loader == "odb" else None,
            "join": args.join if args.loader == "odb" else None,
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"train_summary_{args.loader}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        accelerator.print("[odb-mm-mix-summary] " + json.dumps(summary, sort_keys=True))
        if args.save_final_model:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    cleanup_after_training(accelerator, train_loader, model, optimizer)
    if accelerator.is_main_process:
        print("[odb-mm-mix-summary] teardown_complete=true", flush=True)
    exit_successfully_after_teardown()


if __name__ == "__main__":
    main()
