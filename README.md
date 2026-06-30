# Online Dynamic Batching for Accelerate

Train and evaluate a public multimodal fine-tuning example with
[Online Dynamic Batching](https://github.com/online-dynamic-batching/online-dynamic-batching)
and Hugging Face Accelerate.

This repository is a runnable integration example. It is not a reproduction
package for the paper's experimental numbers; throughput and quality metrics
can differ with hardware, storage, dataset composition, model checkpoints, and
software versions.

## Prerequisites

- A Python environment with PyTorch and NVIDIA GPU support.
- A local or Hugging Face-accessible Qwen3-VL-2B-Instruct checkpoint, provided
  through `ODB_MM_MIX_MODEL` when you do not want to use the default model id.
- Network access to GitHub and the public data/model sources, or equivalent
  local mirrors.
- Enough disk space for the generated public TMDB data, checkpoints, validation
  outputs, and MMMU-MC benchmark outputs.

## Run ODB

Use a Python environment with PyTorch/GPU support, then run:

```bash
export ODB_MM_MIX_MODEL=/path/to/Qwen3-VL-2B-Instruct
./run.sh all-odb
```

This installs the example dependencies, builds the public data, trains the
Accelerate ODB path, and runs validation loss plus MMMU-MC evaluation.

By default training uses one process. Set `ODB_MM_MIX_NUM_PROCESSES=8` for an
8-GPU Accelerate run. The default training run uses a small public subset so the
example finishes quickly; set `ODB_MM_MIX_TRAIN_SIZE=0` to use the full public
training split.

## Tested Workflow

The tested workflow uses `online-dynamic-batching>=0.1.2`, Qwen3-VL-2B-Instruct,
the public MM-Mix TMDB recipe, and the LLaMA-Factory-compatible validation
split (`val_size=0.05`, `split_seed=42`). It covers:

- `./run.sh all-odb`: data build, ODB training, validation loss, and MMMU-MC.
- `./run.sh train-standard` plus `./run.sh eval-standard`: fixed-batch
  baseline training and evaluation.

The records under [results/](results/) are example run records.
They are useful for checking that the example behaves sensibly, but they should
not be read as paper-number reproduction results.

The final full-epoch ODB path was also checked for clean process teardown after
training and checkpoint save. For stable benchmark runs, pre-cache MMMU locally
and run evaluation with `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

## Run Step By Step

```bash
# Install ODB and the helper dependencies for this example.
./run.sh install

# Download/build the public multimodal TMDB training data.
./run.sh data

# Train the ODB run and save the final checkpoint for evaluation.
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh train-odb

# Compute validation loss and MMMU-MC for the ODB checkpoint.
./run.sh eval-odb
```

The default paths are:

- Public data: `data/mm-mix-tmdb`
- Dataset builder checkout: `.deps/build-mm-mix-dataset`
- Checkpoints and eval outputs: `outputs/accelerate-real`

## Run Standard

After `./run.sh install` and `./run.sh data`, run the fixed-batch baseline:

```bash
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh train-standard
./run.sh eval-standard
```

## Common Options

```bash
# Use 8 GPUs through Accelerate.
ODB_MM_MIX_NUM_PROCESSES=8 ./run.sh train-odb

# Pick a different launcher port when running multiple jobs on one machine.
ODB_MM_MIX_MASTER_PORT=29682 ./run.sh train-odb

# Use the full public training split.
ODB_MM_MIX_TRAIN_SIZE=0 ./run.sh train-odb

# Save a checkpoint for validation loss and benchmark evaluation.
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh train-odb

# Tune the image cap for larger or smaller visual inputs.
ODB_MM_MIX_IMAGE_MAX_PIXELS=589824 ./run.sh train-odb

# Evaluate a custom checkpoint.
ODB_ACCEL_EVAL_CHECKPOINT=/path/to/checkpoint ./run.sh eval-valloss
ODB_ACCEL_EVAL_CHECKPOINT=/path/to/checkpoint ./run.sh benchmark
```

## Outputs

Default model directories:

| Target | Directory |
| --- | --- |
| ODB | `outputs/accelerate-real/odb` |
| Standard | `outputs/accelerate-real/standard` |

Validation-loss outputs are written under the evaluated checkpoint directory as
`eval_out_accelerate_valloss`.

MMMU-MC outputs are written under the evaluated checkpoint directory as
`mmmu_mc_likelihood_accelerate` and include:

- `mmmu_mc_likelihood_results.json`
- `predictions.jsonl`
- `excluded.jsonl`
- `score_audit.json`

## Commands

| Command | Purpose |
| --- | --- |
| `./run.sh install` | Install Python dependencies for this example. |
| `./run.sh data` | Build the public TMDB data. |
| `./run.sh train-odb` | Train with ODB. |
| `./run.sh eval-odb` | Evaluate the ODB checkpoint. |
| `./run.sh train-standard` | Train the fixed-batch baseline. |
| `./run.sh eval-standard` | Evaluate the Standard checkpoint. |
| `./run.sh eval-valloss` | Evaluate validation loss for a saved checkpoint. |
| `./run.sh benchmark` | Run the built-in MMMU-MC benchmark. |
| `./run.sh all-odb` | Run the complete ODB path. |

## Manual Training Launcher

Use `scripts/train_accelerate.py` when you need to override
individual knobs without editing `run.sh`:

```bash
accelerate launch --num_processes 8 scripts/train_accelerate.py \
  --loader odb \
  --data data/mm-mix-tmdb \
  --model "$ODB_MM_MIX_MODEL" \
  --token-budget 12288 \
  --buffer-size 1024 \
  --loss-scaling exact \
  --join \
  --deepspeed-config configs/ds_z2.json \
  --train-size 128 \
  --max-steps 0
```

Set `--train-size 0` for the full public training split. Use `--loader
standard` for the fixed-batch baseline.

## Integration Notes

This example uses a native Accelerate training loop. The data path is:

```text
public MM-Mix TMDB
  -> lazy Dataset returns one processed tensor sample per __getitem__
  -> ODB groups real post-processor lengths
  -> tensor-only collator pads/stacks each ODB group
  -> Accelerate training loop runs model.forward / loss / optimizer
```

For a model-specific integration, keep the same boundary: processor output
must be a single-sample tensor dictionary before ODB grouping.

For the package API contract behind this example, see the
[Accelerate integration guide](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/accelerate.md).

The default image cap is chosen for stable out-of-the-box execution. Tune
`ODB_MM_MIX_IMAGE_MAX_PIXELS` when you want to allow larger images.

## Related Examples

- Shared dataset builder: [build-mm-mix-dataset](https://github.com/online-dynamic-batching/build-mm-mix-dataset)
- LLaMA-Factory example: [odb-example-llamafactory](https://github.com/online-dynamic-batching/odb-example-llamafactory)
- HF Trainer example: [odb-example-hf-trainer](https://github.com/online-dynamic-batching/odb-example-hf-trainer)
- Lightning example: [odb-example-lightning](https://github.com/online-dynamic-batching/odb-example-lightning)
