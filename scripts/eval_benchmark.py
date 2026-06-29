#!/usr/bin/env python3
"""Evaluate a saved Accelerate MM-Mix checkpoint on MMMU-MC.

This script is intentionally self-contained for the public example. It runs a
parser-free choice-likelihood benchmark: for each MMMU multiple-choice sample,
it scores the next-token log-probability of answer letters A-H and picks the
highest-scoring valid option letter.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor


PROTOCOL_NAME = "MMMU-MC-choice-likelihood-v1"
PROMPT_SUFFIX = "Answer with the option letter only."
CANDIDATE_LETTERS = tuple("ABCDEFGH")
QWEN3VL_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
}
QWEN3VL_MROPE_SECTION = [24, 20, 20]

MMMU_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

CATEGORY_MAP = {
    "Art": "Art & Design",
    "Art_Theory": "Art & Design",
    "Design": "Art & Design",
    "Music": "Art & Design",
    "Accounting": "Business",
    "Economics": "Business",
    "Finance": "Business",
    "Manage": "Business",
    "Marketing": "Business",
    "Biology": "Science",
    "Chemistry": "Science",
    "Geography": "Science",
    "Math": "Science",
    "Physics": "Science",
    "Basic_Medical_Science": "Health & Medicine",
    "Clinical_Medicine": "Health & Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health & Medicine",
    "Pharmacy": "Health & Medicine",
    "Public_Health": "Health & Medicine",
    "History": "Humanities & Social Science",
    "Literature": "Humanities & Social Science",
    "Psychology": "Humanities & Social Science",
    "Sociology": "Humanities & Social Science",
    "Agriculture": "Tech & Engineering",
    "Architecture_and_Engineering": "Tech & Engineering",
    "Computer_Science": "Tech & Engineering",
    "Electronics": "Tech & Engineering",
    "Energy_and_Power": "Tech & Engineering",
    "Materials": "Tech & Engineering",
    "Mechanical_Engineering": "Tech & Engineering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=os.getenv("ODB_ACCEL_EVAL_CHECKPOINT")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.getenv("ODB_ACCEL_BENCHMARK_OUTPUT_DIR")
        or os.getenv("ODB_ACCEL_EVAL_SAVE_DIR"),
    )
    parser.add_argument(
        "--dataset", default=os.getenv("ODB_ACCEL_BENCHMARK_DATASET", "MMMU/MMMU")
    )
    parser.add_argument(
        "--split", default=os.getenv("ODB_ACCEL_BENCHMARK_SPLIT", "validation")
    )
    parser.add_argument("--subjects", default=os.getenv("ODB_ACCEL_EVAL_SUBJECTS"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("ODB_ACCEL_EVAL_MAX_SAMPLES", "0")),
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.getenv("ODB_ACCEL_EVAL_TORCH_DTYPE", "bf16"),
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=os.getenv("HF_DATASETS_CACHE")
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def patch_qwen3vl_rope_scaling(config):
    candidates = [config]
    for name in ("text_config", "vision_config", "llm_config", "language_config"):
        child = getattr(config, name, None)
        if child is not None and child not in candidates:
            candidates.append(child)
    if not any("qwen3_vl" in str(getattr(cfg, "model_type", "")) for cfg in candidates):
        return config

    for cfg in candidates:
        if hasattr(cfg, "rope_scaling") and getattr(cfg, "rope_scaling", None) is None:
            cfg.rope_scaling = {
                "rope_type": "default",
                "mrope_section": list(QWEN3VL_MROPE_SECTION),
            }
    return config


def load_processor(checkpoint: Path, *, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
    except AttributeError as exc:
        if "keys" not in str(exc):
            raise
        return AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
            extra_special_tokens=QWEN3VL_EXTRA_SPECIAL_TOKENS,
        )


def load_model(checkpoint: Path, *, dtype: torch.dtype, trust_remote_code: bool):
    import transformers

    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    config = patch_qwen3vl_rope_scaling(
        AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    )
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = model_cls.from_pretrained(checkpoint, **kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model


def model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def get_mmmu_options(item: dict[str, Any]) -> list[str]:
    raw = item.get("options")
    if raw is not None:
        parsed = raw
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple)):
            options = [str(option).strip() for option in parsed if str(option).strip()]
            if options:
                return options
    return [
        str(item.get(f"option_{chr(65 + i)}", "")).strip()
        for i in range(8)
        if str(item.get(f"option_{chr(65 + i)}", "")).strip()
    ]


def collect_images(item: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for i in range(1, 8):
        value = item.get(f"image_{i}")
        if value is None:
            continue
        if isinstance(value, Image.Image):
            images.append(value.convert("RGB"))
        elif isinstance(value, str) and os.path.exists(value):
            with Image.open(value) as opened:
                images.append(opened.convert("RGB"))
    return images


def format_prompt(question: str, options: list[str]) -> str:
    option_text = "\n".join(
        f"({chr(65 + i)}) {option}" for i, option in enumerate(options)
    )
    return f"{question}\n{option_text}\n{PROMPT_SUFFIX}"


def candidate_token_info(processor) -> dict[str, dict[str, Any]]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError(
            "processor has no tokenizer; cannot score letter candidates"
        )
    info: dict[str, dict[str, Any]] = {}
    for letter in CANDIDATE_LETTERS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"candidate {letter!r} produced no token ids")
        info[letter] = {
            "candidate_text": letter,
            "first_token_id": int(token_ids[0]),
            "token_ids": [int(token_id) for token_id in token_ids],
        }
    return info


def build_inputs(
    processor, prompt: str, images: list[Image.Image], device: torch.device
):
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    return inputs.to(device), text


def score_candidate_logprobs(
    model,
    processor,
    prompt: str,
    images: list[Image.Image],
    token_info: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    device = model_input_device(model)
    inputs, rendered_chat = build_inputs(processor, prompt, images, device)
    with torch.inference_mode():
        try:
            outputs = model(**inputs, use_cache=False)
        except TypeError as exc:
            if "use_cache" not in str(exc):
                raise
            outputs = model(**inputs)

    logits = outputs.logits
    attention_mask = inputs.get("attention_mask")
    last_idx = (
        int(attention_mask[0].sum().item()) - 1
        if attention_mask is not None
        else logits.shape[1] - 1
    )
    logprobs = torch.log_softmax(logits[0, last_idx, :].float(), dim=-1)
    scores = {
        letter: float(logprobs[int(meta["first_token_id"])].item())
        for letter, meta in token_info.items()
    }
    return scores, {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "rendered_chat_chars": len(rendered_chat),
    }


def validate_letter_mc_item(
    item: dict[str, Any],
) -> tuple[bool, str, str, list[str], list[str], str]:
    answer = str(item.get("answer", "")).strip().upper()
    options = get_mmmu_options(item)
    if not re.fullmatch(r"[A-H]", answer):
        return False, "non_letter_answer", answer, options, [], "non_letter_answer"
    answer_idx = ord(answer) - ord("A")
    if not options:
        return (
            True,
            "",
            answer,
            options,
            list(CANDIDATE_LETTERS),
            "missing_options_scored_all_letters",
        )
    if len(options) > len(CANDIDATE_LETTERS):
        return (
            True,
            "",
            answer,
            options,
            list(CANDIDATE_LETTERS),
            "too_many_options_scored_AH",
        )
    if answer_idx >= len(options):
        return (
            True,
            "",
            answer,
            options,
            list(CANDIDATE_LETTERS),
            "answer_outside_options_scored_all_letters",
        )
    return (
        True,
        "",
        answer,
        options,
        list(CANDIDATE_LETTERS[: len(options)]),
        "options_valid",
    )


def sample_id(item: dict[str, Any], subject: str, index: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"{subject}:{index}"


def choose_prediction(
    scores: dict[str, float], valid_letters: list[str]
) -> tuple[str, float | None, dict[str, float]]:
    choice_scores = {letter: scores[letter] for letter in valid_letters}
    ranked = sorted(choice_scores.items(), key=lambda item: (-item[1], item[0]))
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    return ranked[0][0], margin, choice_scores


def parse_subjects(raw: str | None) -> list[str]:
    if not raw:
        return list(MMMU_SUBJECTS)
    subjects = [subject.strip() for subject in raw.split(",") if subject.strip()]
    unknown = [subject for subject in subjects if subject not in MMMU_SUBJECTS]
    if unknown:
        raise ValueError(f"unknown MMMU subjects: {unknown}")
    return subjects


def summarize_accuracy(values: list[bool]) -> dict[str, Any]:
    correct = int(sum(values))
    total = len(values)
    accuracy = correct / total * 100.0 if total else 0.0
    return {"accuracy": round(accuracy, 2), "total": total, "correct": correct}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        raise SystemExit("--checkpoint or ODB_ACCEL_EVAL_CHECKPOINT is required")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_dir = args.output_dir or (checkpoint / "mmmu_mc_likelihood_accelerate")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(checkpoint, trust_remote_code=args.trust_remote_code)
    model = load_model(
        checkpoint,
        dtype=torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    token_info = candidate_token_info(processor)
    subjects = parse_subjects(args.subjects)

    predictions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_by_reason: Counter[str] = Counter()
    option_status_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    category_correct: dict[str, list[bool]] = defaultdict(list)
    total_rows_seen = 0
    total_images = 0
    score_meta_examples: list[dict[str, Any]] = []

    stop = False
    for subject in tqdm(subjects, desc="Subjects"):
        dataset = load_dataset(
            args.dataset,
            subject,
            split=args.split,
            trust_remote_code=True,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        subject_correct: list[bool] = []
        subject_excluded = 0
        for index, raw_item in enumerate(
            tqdm(dataset, desc=f"  {subject}", leave=False)
        ):
            item = dict(raw_item)
            total_rows_seen += 1
            sid = sample_id(item, subject, index)
            ok, reason, answer, options, valid_letters, option_status = (
                validate_letter_mc_item(item)
            )
            if not ok:
                subject_excluded += 1
                excluded_by_reason[reason] += 1
                excluded.append(
                    {
                        "protocol": PROTOCOL_NAME,
                        "split": args.split,
                        "subject": subject,
                        "sample_id": sid,
                        "reason": reason,
                        "answer_raw": str(item.get("answer", "")),
                        "normalized_answer": answer,
                        "options": options,
                    }
                )
                continue

            images = collect_images(item)
            prompt = format_prompt(str(item.get("question", "")), options)
            scores, score_meta = score_candidate_logprobs(
                model, processor, prompt, images, token_info
            )
            prediction, margin, choice_scores = choose_prediction(scores, valid_letters)
            correct = prediction == answer
            prediction_counts[prediction] += 1
            option_status_counts[option_status] += 1
            total_images += len(images)
            if len(score_meta_examples) < 5:
                score_meta_examples.append(
                    score_meta | {"subject": subject, "sample_id": sid}
                )

            category = CATEGORY_MAP.get(subject, "Other")
            subject_correct.append(bool(correct))
            category_correct[category].append(bool(correct))
            category_correct["Overall"].append(bool(correct))
            predictions.append(
                {
                    "protocol": PROTOCOL_NAME,
                    "split": args.split,
                    "subject": subject,
                    "category": category,
                    "sample_id": sid,
                    "question": item.get("question", ""),
                    "options": options,
                    "valid_candidate_letters": valid_letters,
                    "option_status": option_status,
                    "ground_truth": answer,
                    "candidate_scores": scores,
                    "choice_scores": choice_scores,
                    "prediction": prediction,
                    "score_margin": margin,
                    "correct": bool(correct),
                    "num_images": len(images),
                }
            )
            if args.max_samples > 0 and len(predictions) >= args.max_samples:
                stop = True
                break

        summary = summarize_accuracy(subject_correct)
        print(
            f"  {subject}: {summary['accuracy']:.2f}% "
            f"({summary['correct']}/{summary['total']}), excluded={subject_excluded}",
            flush=True,
        )
        if stop:
            break

    overall = category_correct["Overall"]
    if not overall:
        write_jsonl(output_dir / "excluded.jsonl", excluded)
        raise SystemExit("No MMMU-MC samples were evaluated.")

    overall_summary = summarize_accuracy(overall)
    per_category = {
        category: summarize_accuracy(values)
        for category, values in category_correct.items()
        if values
    }
    results = {
        "protocol": PROTOCOL_NAME,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "dataset": args.dataset,
        "subjects": subjects,
        "evaluated_samples": len(predictions),
        "excluded_samples": len(excluded),
        "total_rows_seen": total_rows_seen,
        "overall_accuracy": overall_summary["accuracy"],
        "overall_correct": overall_summary["correct"],
        "overall_total": overall_summary["total"],
        "per_category": per_category,
        "excluded_by_reason": dict(excluded_by_reason),
        "prediction_counts": dict(prediction_counts),
        "option_status_counts": dict(option_status_counts),
        "total_images": total_images,
    }
    audit = {
        "protocol": PROTOCOL_NAME,
        "candidate_token_info": token_info,
        "prompt_suffix": PROMPT_SUFFIX,
        "score_meta_examples": score_meta_examples,
    }

    (output_dir / "mmmu_mc_likelihood_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "score_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "excluded.jsonl", excluded)
    print(
        f"[odb-hf-benchmark] overall_accuracy={overall_summary['accuracy']:.2f} "
        f"({overall_summary['correct']}/{overall_summary['total']}) result={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
