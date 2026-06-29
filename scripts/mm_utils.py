"""Small helpers shared by Accelerate MM-Mix examples."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from odb_mm_mix import collate_tokens
import torch


VISION_SPECIAL_TOKENS = (
    "<image>",
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|video_pad|>",
)


def collect_vision_token_ids(processor: Any) -> set[int]:
    """Collect common vision special-token ids for label masking checks."""
    ids: set[int] = set()
    value = getattr(processor, "image_token_id", None)
    if value is not None:
        ids.add(int(value))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for token in VISION_SPECIAL_TOKENS:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and token_id != getattr(tokenizer, "unk_token_id", None)
            ):
                ids.add(token_id)
    return ids


def mask_vision_tokens(
    batch: dict[str, Any], vision_token_ids: set[int]
) -> dict[str, Any]:
    """Mask known vision special tokens in labels if they are present."""
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return batch
    for token_id in vision_token_ids:
        labels[input_ids == int(token_id)] = -100
    return batch


def make_model_collator(
    processor: Any,
    *,
    compute_dtype: torch.dtype | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return a collator whose output can be passed directly to HF VLM models."""
    vision_token_ids = collect_vision_token_ids(processor)

    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = collate_tokens(rows)
        batch.pop("odb_n_patches", None)
        batch = mask_vision_tokens(batch, vision_token_ids)
        if compute_dtype is not None:
            for key, value in list(batch.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    batch[key] = value.to(compute_dtype)
        return batch

    return _collate


def _resolve_rope_func(model: Any) -> Callable[..., Any] | None:
    candidates = [model]
    seen: set[int] = set()
    index = 0
    while index < len(candidates):
        root = candidates[index]
        index += 1
        if id(root) in seen:
            continue
        seen.add(id(root))
        rope_func = getattr(root, "get_rope_index", None)
        if callable(rope_func):
            return rope_func
        for attr in ("module", "base_model", "model"):
            child = getattr(root, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)
    return None


def _model_type(model: Any) -> str | None:
    candidates = [model]
    seen: set[int] = set()
    index = 0
    while index < len(candidates):
        root = candidates[index]
        index += 1
        if id(root) in seen:
            continue
        seen.add(id(root))
        for attr in ("module", "base_model", "model"):
            child = getattr(root, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        value = getattr(config, "model_type", None)
        if value is not None:
            return str(value)
    return None


def reset_qwen_vl_rope_cache(model: Any) -> None:
    """Clear Qwen-VL rope cache before/after each variable-shape forward."""
    candidates = [model]
    for root in list(candidates):
        for attr in ("module", "base_model", "model"):
            child = getattr(root, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)
    for candidate in candidates:
        inner = getattr(candidate, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        if hasattr(candidate, "rope_deltas"):
            candidate.rope_deltas = None


def add_qwen_vl_position_ids(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    """Match LLaMA-Factory's Qwen-VL mROPE position-id preparation."""
    if not isinstance(batch, dict) or "position_ids" in batch:
        return batch
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        return batch

    rope_func = _resolve_rope_func(model)
    model_type = _model_type(model)
    if rope_func is None:
        if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5"}:
            raise ValueError(f"{model_type} requires get_rope_index for mROPE position ids.")
        return batch

    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "image_grid_thw": batch.get("image_grid_thw"),
        "video_grid_thw": batch.get("video_grid_thw"),
        "attention_mask": (attention_mask >= 1).float(),
    }
    signature = inspect.signature(rope_func)
    if "mm_token_type_ids" in signature.parameters:
        mm_token_type_ids = batch.get("mm_token_type_ids")
        if isinstance(mm_token_type_ids, torch.Tensor):
            kwargs["mm_token_type_ids"] = mm_token_type_ids
    if "second_per_grid_ts" in batch:
        kwargs["second_per_grid_ts"] = batch.get("second_per_grid_ts")
    elif "video_second_per_grid" in batch:
        kwargs["second_per_grids"] = batch.get("video_second_per_grid")

    position_ids, rope_deltas = rope_func(**kwargs)
    batch["position_ids"] = position_ids
    batch["rope_deltas"] = rope_deltas
    batch.pop("mm_token_type_ids", None)
    return batch
