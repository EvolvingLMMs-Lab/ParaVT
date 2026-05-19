# Evaluation

One driver (`paravt/eval/driver.py`) covers the seven long-video benchmarks reported in the paper across the seven native prompt protocols used by each baseline class. The launcher (`scripts/run_eval.sh`) iterates each recipe's `datasets:` list one at a time, killing vLLM between iterations so each dataset starts from a clean session.

## Environment

```bash
bash scripts/setup_env.sh eval        # creates .venv-eval from requirements/eval.lock
source .venv-eval/bin/activate
```

The eval venv is independent from `.venv-sft` and `.venv-rl` because vLLM (eval) and SGLang (RL) pin different torch builds. `requirements/eval.lock` covers vLLM 0.17.x + torch 2.10.x + transformers + qwen-vl-utils + datasets.

## Quick start

```bash
# A single row (ParaVT-8B headline reproduce)
PARAVT_EVAL_MODEL=ParaVT/ParaVT-8B \
    bash paravt/eval/scripts/reproduce_paravt_8b.sh

# Every row in sequence on one 8-GPU box
bash paravt/eval/scripts/batch_reproduce_tab1.sh
```

Per-dataset JSON lands at `${PARAVT_EVAL_OUT}/<row>/<channel>/<dataset>.json` and is rolled up into `summary_all.json`. Re-runs are resume-friendly: the driver skips datasets whose summary is already on disk. Each summary carries `accuracy` (or `mIoU` / `R@{0.3,0.5,0.7}` for Charades-STA), `avg_tool_calls`, and `tool_usage_rate`, plus per-duration / per-config breakdowns where applicable.

## Benchmark splits

The driver pulls every benchmark from the HuggingFace Hub via `datasets`; no manual download step is required. The seven splits the headline table reports on:

| `--datasets` value | HF dataset id | Split |
|---|---|---|
| `videomme` | [`lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME) | `test`, w/o subtitles |
| `videomme_wsub` | [`lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME) | `test`, w/ subtitles |
| `longvideobench` | [`longvideobench/LongVideoBench`](https://huggingface.co/datasets/longvideobench/LongVideoBench) | `validation` |
| `lvbench` | [`lmms-lab/LVBench`](https://huggingface.co/datasets/lmms-lab/LVBench) | `test` |
| `mlvu` | [`sy1998/MLVU_dev`](https://huggingface.co/datasets/sy1998/MLVU_dev) | `dev` |
| `mmvu` | [`yale-nlp/MMVU`](https://huggingface.co/datasets/yale-nlp/MMVU) | `validation` |
| `charades` | [`lmms-lab/charades_sta`](https://huggingface.co/datasets/lmms-lab/charades_sta) | `test` |

Two splits ship videos out-of-band and need an explicit local path:

- **MLVU** — `sy1998/MLVU_dev` ships `video_part_*.zip`; extract once into the directory pointed at by `PARAVT_MLVU_VIDEO_DIR` (defaults to `$HF_HOME/MLVU/video`).
- **MMVU** — first eval triggers a ~30 GB `snapshot_download` under `$HF_HOME/hub`; override with `PARAVT_MMVU_VIDEO_DIR` if you have a mirrored copy.

Both env vars are documented in [`.secrets.env.example`](../../.secrets.env.example).

## CLI flags & locked-in protocol

```bash
python -m paravt.eval.driver \
    --model_path <hf_id_or_local_snapshot> \
    --datasets <csv> \
    --output_dir <dir> \
    [--prompt_mode <mode>] [--video_channel <channel>] [...]
```

| Flag | Default | Purpose |
|---|---|---|
| `--model_path` | required | HF Hub id or local snapshot |
| `--datasets` | required | csv from the Benchmark splits table |
| `--output_dir` | required | output root; per-dataset JSON lands at `<output_dir>/<dataset>.json` |
| `--prompt_mode` | `agentic_general` | one of the modes in the next section |
| `--video_channel` | `image_url` | `image_url` (client-side base64 frames) or `video_url` (vLLM server-side decode) |
| `--no_mm_kwargs` | ON | skip vLLM `--mm-processor-kwargs`; pass `--mm_kwargs` to re-enable |
| `--main_nframes` | 64 | overview frame budget |
| `--max_turns` | 3 | tool-dialog cap (agentic modes only) |
| `--max_parallel` | 5 | per-turn tool-call cap |
| `--max_tokens` | 2048 | per-completion output cap |
| `--num_gpus` | 8 | vLLM dp replicas |
| `--max_num_batched_tokens` | 131072 | vLLM prefill ceiling |
| `--reuse_servers` | off | reuse a pre-launched vLLM session |
| `--shard_id` / `--num_shards` | 0 / 1 | cross-machine slicing |
| `--smoke_test` | off | 1 GPU, 5 samples per dataset |
| `--limit N` | none | finer-grained dev-loop sample cap |

The values below reproduce the paper's headline row; do not change them without re-running the full sweep:

| Knob | Value | Where it lives |
|---|---|---|
| `nframes` | 64 | `--main_nframes 64` |
| temperature | 0 (greedy) | `paravt/eval/driver.py:_completion_with_retry` |
| `max_tokens` | 2048 | `--max_tokens 2048` |
| force-answer `max_tokens` | 64 (MCQ) / 128 (charades) | hard-coded in `eval_one` |
| `max_turns` | 3 | recipe + driver default |
| `max_parallel` | 5 | recipe + driver default |
| MCQ channel | `image_url` | `--video_channel image_url` |
| charades channel | `video_url` | `--video_channel video_url` |
| `--mm-processor-kwargs` | DISABLED | `--no_mm_kwargs` (default ON) |
| `--max-num-batched-tokens` | 131072 | driver default |
| `--limit-mm-per-prompt` | `{"video":1,"image":1024}` | hard-coded in `launch_servers` |
| vLLM | 0.17.x | `requirements/eval.lock` |

## Prompt modes

The driver pins one `--prompt_mode` per baseline class because running the wrong prompt at evaluation time silently degrades that row's number; there is no automatic fallback. The seven prompt strings live verbatim in `paravt/eval/driver.py`.

| `--prompt_mode` | Used for | Native protocol | Reproduce script(s) |
|---|---|---|---|
| `direct` | Qwen2.5-VL-7B | base instruct, no native think pattern | `reproduce_qwen25vl_7b.sh` |
| `reasoning` | Video-R1, VideoRFT, VideoChat-R1, Video-Thinker, Time-R1, ReWatch-R1 | `<think>...</think><answer>...</answer>` verbatim from each baseline's card | `reproduce_reasoning_baselines.sh` |
| `agentic_general` | Qwen3-VL-8B (tool), Conan-7B, **ParaVT-8B (Ours)** | parallel `<tool_call>` dispatch with explicit Workflow / Important / Format scaffolding | `reproduce_qwen3vl_8b.sh`, `reproduce_conan_7b.sh`, `reproduce_paravt_8b.sh` |
| `agentic_minimal` | **ParaVT-8B** on Charades-STA only | bare Hermes-2 tool-call envelope; tighter temporal spans than the full agentic envelope | sub-call inside `reproduce_paravt_8b.sh` |
| `agentic_videozoomer` | VideoZoomer-7B | native `<video_zoom>` tool protocol (placeholder, see below) | `reproduce_videozoomer_7b.sh` |
| `agentic_sage` | SAGE-7B | Context-VLM JSON 5-tool schema (placeholder, see below) | `reproduce_sage_7b.sh` |
| `agentic_longvt` | LongVT-RFT-7B | iMCoTT prompt suffix from `lmms_eval_tasks/videomme/utils.py` (placeholder, see below) | `reproduce_longvt_rft.sh` |

Model paths used by each reproduce script:

| Row | `--model_path` |
|---|---|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen3-VL-8B (tool) | `Qwen/Qwen3-VL-8B-Instruct` |
| Video-R1-7B | `Video-R1/Video-R1-7B` |
| VideoRFT-7B | `QiWang98/VideoRFT` |
| VideoChat-R1-7B | `OpenGVLab/VideoChat-R1_7B` |
| Video-Thinker-7B | `ShijianW01/Video-Thinker-7B` |
| Time-R1-7B | `Boshenxx/Time-R1-7B` |
| ReWatch-R1-7B | `zcccccz/ReWatch-R1` |
| Conan-7B | `RUBBISHLIKE/Conan-7B` |
| VideoZoomer-7B | `zsgvivo/videozoomer` |
| SAGE-7B | `allenai/SAGE-MM-Qwen2.5-VL-7B-SFT_RL` |
| LongVT-RFT-7B | `longvideotool/LongVT-RFT` |
| **ParaVT-8B (Ours)** | `ParaVT/ParaVT-8B` |

### Filling baseline-specific placeholders

Three modes ship as runtime-guarded placeholders because their native protocols sit outside this release tree:

```python
# paravt/eval/driver.py
VIDEOZOOMER_AGENTIC_SYSTEM = "FILL_FROM_VIDEOZOOMER"
SAGE_AGENTIC_SYSTEM        = "FILL_FROM_SAGE"
LONGVT_TOOL_PROMPT_SUFFIX  = "FILL_FROM_LONGVT"
```

Selecting one before pasting in the prompt body raises a runtime `ValueError`. To enable a placeholder row:

1. **VideoZoomer** — paste the native `<video_zoom>` system prompt into `VIDEOZOOMER_AGENTIC_SYSTEM`. The driver already ships `parse_video_zoom_calls()` for the response shape.
2. **SAGE** — paste SAGE-7B's Context-VLM JSON 5-tool schema and system prompt into `SAGE_AGENTIC_SYSTEM`, then extend `parse_tool_calls` to recognize SAGE's tool names (or add a SAGE-specific parser following the `parse_video_zoom_calls` precedent).
3. **LongVT** — paste the verbatim `TOOL_PROMPT` suffix from `EvolvingLMMs-Lab/LongVT/lmms_eval_tasks/videomme/utils.py` into `LONGVT_TOOL_PROMPT_SUFFIX`. The dispatch loop is shared with `agentic_general` because LongVT-RFT emits the same `<tool_call>` JSON form.

Once filled, set `PARAVT_RUN_STUBS=1` and `paravt/eval/scripts/batch_reproduce_tab1.sh` will pick them up.

## Running one dataset at a time

Streaming a single vLLM session through five or more long-video benchmarks back-to-back is empirically unsafe: connection drop rates climb sharply on later datasets, and the driver's per-dataset error counter does **not** catch mid-call connection drops, so the result is silently degraded numbers. `scripts/run_eval.sh` kills the vLLM servers between datasets and lets the next dataset start from a clean state. Every per-row reproduce script under `paravt/eval/scripts/` calls into it.
