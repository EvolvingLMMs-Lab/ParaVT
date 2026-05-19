# lmms-engine patches

ParaVT vendors a copy of [`EvolvingLMMs-Lab/lmms-engine`](https://github.com/EvolvingLMMs-Lab/lmms-engine) at commit **`e445004`** under `paravt/sft/lmms-engine/` at the repository root. The patches in this directory document the diffs we maintain on top of that pin so reviewers and downstream forks can see exactly what changed and why.

If you instead choose to install lmms-engine from upstream (`pip install lmms-engine` or a fresh clone) you will need to apply these patches by hand. The vendored copy already has them applied.

## Pin

| Repo | Commit | Date |
|---|---|---|
| `EvolvingLMMs-Lab/lmms-engine` | `e445004957816e9cd1edd66fb36d1990af66a514` | 2026-04-07 |

## Patches

### `0001-data-utils-schema-align.patch`

**Problem.** `concatenate_datasets()` fails when nested struct schemas differ across input parquets — e.g., one parquet has `image_url: {url: string}` while another infers `image_url: null` because every value happens to be null. lmms-engine surfaces this as a HuggingFace `datasets` schema-merge error during the first epoch.

**Root cause.** Parquet's automatic schema inference produces incompatible types when a column is uniformly null in one shard but populated in another.

**Fix.** Replace `concatenate_datasets(data_list)` with `_align_and_concatenate(data_list)`, which materializes every dataset to a list of dicts first and then re-creates a single unified `Dataset` with a consistent schema.

**Upstream candidate fix.** Cast all datasets to a common schema before concatenation, or switch to `interleave_datasets`.

### `0002-qwen3vl-iterable-messages-as-json.patch`

**Problem.** When the `messages` column is stored as a JSON string (a common workaround for parquet's nested-schema issues), lmms-engine's loader crashes because it expects a list/dict.

**Fix.** Decode the JSON-string form lazily in `load_from_json()`:

```python
if isinstance(messages, str):
    messages = json.loads(messages)
```

**Upstream candidate fix.** Native support for JSON-string-encoded messages in the loader.

### `0003-vision-iterable-messages-as-json.patch`

Same bug as **#2**, applied to `vision_iterable_dataset.py` for the parallel non-Qwen-3-VL code path.

## Dependency note

`requirements/sft.lock` pins `qwen-vl-utils==0.0.14`. Versions ≤ 0.0.11 lack the `return_video_metadata` parameter the processor pipeline depends on; this is a transitive dependency change, not an lmms-engine source patch.

## Release-side trim

The vendored `paravt/sft/lmms-engine/` drops upstream `docs/` (Sphinx sources) and `examples/` (per-model training scripts), neither of which is exercised by the ParaVT pipeline. The full upstream tree is preserved on the `paravt-full` branch and at https://github.com/EvolvingLMMs-Lab/lmms-engine.
