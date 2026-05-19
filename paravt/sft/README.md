# SFT cold-start

The PARA-GRPO recipe is initialized from a Qwen3-VL-8B SFT checkpoint trained on a tool-use mix. This directory holds the recipe that produces that checkpoint.

| File | Purpose |
|---|---|
| `qwen3vl_8b.yaml` | Default SFT recipe — packing 8192, FSDP2, lr 2e-5 cosine |
| `data_manifest.example.yaml` | Sample data manifest listing the parquet shards that compose the SFT mix; replace the `${PARAVT_DATA_ROOT}` placeholders with absolute paths |

## Environment

```bash
bash scripts/setup_env.sh sft         # creates .venv-sft from requirements/sft.lock
source .venv-sft/bin/activate
```

The SFT venv runs lmms-engine on FSDP2; it is independent of the RL and eval venvs because flash-attn / liger-kernel pins disagree across them. lmms-engine is vendored under `paravt/sft/lmms-engine/` with our patches already applied (see `patches/lmms-engine/` for what we changed vs. upstream).

## Reproducing the SFT cold-start

```bash
export PARAVT_BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
export PARAVT_FILEROOT=./experiments
export PARAVT_DATA_ROOT=/path/to/your/parquet/root
export PARAVT_SFT_DATA=paravt/sft/configs/data_manifest.example.yaml

bash scripts/run_sft.sh paravt/sft/configs/qwen3vl_8b.yaml
```

Checkpoints land under `${PARAVT_FILEROOT}/sft-paravt/`. Pass that directory to the RL stage via `PARAVT_BASE_MODEL` (see `paravt/rl/README.md`). The 8 B base on 8 × 80 GB+ NVIDIA GPUs takes ≈ 36 h end-to-end with packing length 8192.

The data manifest lists the SFT parquet shards; the actual corpus is hosted at [`ParaVT/ParaVT-Parquet`](https://huggingface.co/datasets/ParaVT/ParaVT-Parquet) (annotations) and [`ParaVT/ParaVT-Source`](https://huggingface.co/datasets/ParaVT/ParaVT-Source) (source videos). See [`paravt/data/README.md`](../data/README.md) for the `paravt.data.materialize` one-shot script that re-attaches absolute paths.

## Smaller-scale sanity test

For a 1 GPU + tiny-batch dry run that exercises the full launcher (data loading, FSDP2 init, optimizer step, checkpoint save):

```bash
bash scripts/run_sft.sh paravt/sft/configs/qwen3vl_8b.yaml \
    trainer_args.per_device_train_batch_size=1 \
    trainer_args.max_steps=2 \
    trainer_args.save_steps=2 \
    dataset_config.packing_length=4096 \
    dataset_config.video_max_frames=32
```

Two steps land in ≈ 1 minute on a single 80 GB+ NVIDIA GPU; loss + lr + grad_norm + mfu all log to console.

## Key knobs

The SFT recipe is OmegaConf-loaded by `lmms-engine`. Common overrides:

| Knob | Default | When to change |
|---|---|---|
| `trainer_args.per_device_train_batch_size` | 4 | Adjust to your GPU memory |
| `trainer_args.gradient_accumulation_steps` | 1 | Compensate for smaller per-device batch |
| `dataset_config.packing_length` | 8192 | Drop to 4096 for tighter memory budgets |
| `dataset_config.video_max_frames` | 64 | Match RL training (do not lower without retraining downstream) |
| `model_config.attn_implementation` | `flash_attention_2` | `sdpa` if flash-attn is unavailable |

Missing shards are logged and skipped by lmms-engine; the recipe reproduces the published number when every shard listed in `data_manifest.example.yaml` resolves.
