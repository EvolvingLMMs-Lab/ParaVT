# lmms-engine Patches for ParaVT SFT

## 1. Parquet schema alignment issue (data_utils.py)
**Problem**: When loading multiple parquets via YAML config, `concatenate_datasets()` fails if nested struct schemas differ (e.g., some parquets have `image_url: {url: string}` while others infer `image_url: null`).

**Root cause**: Parquet's automatic schema inference produces different types when all values in a column are null vs. when some have values.

**Workaround applied**: Replaced `concatenate_datasets(data_list)` with `_align_and_concatenate(data_list)` which converts all datasets to list-of-dicts first, then creates a single unified dataset.

**Better fix for PR**: Either:
- Cast all datasets to a common schema before concatenation
- Or use `interleave_datasets` which doesn't require schema alignment

**File**: `src/lmms_engine/utils/data_utils.py` line ~107

## 2. Messages as JSON string support (qwen3_vl_iterable_dataset.py)
**Problem**: When `messages` column is stored as JSON string (for parquet schema compatibility), the loader crashes because it expects a list/dict.

**Fix applied**: Added `if isinstance(messages, str): messages = json.loads(messages)` in `load_from_json()`.

**Files**: 
- `src/lmms_engine/datasets/iterable/qwen3_vl_iterable_dataset.py` line 24
- `src/lmms_engine/datasets/iterable/vision_iterable_dataset.py`

**Recommendation for PR**: This is a reasonable feature to support - JSON string encoding avoids all parquet nested schema issues.

## 3. qwen_vl_utils version
**Problem**: `return_video_metadata` parameter not available in qwen_vl_utils 0.0.11.
**Fix**: Upgraded to 0.0.14.

## Training Config Notes
- wandb project: ParaVT (shared with RL runs)
- Base model: Qwen3-VL-8B-Instruct
- System prompt: v2 (matching RL training)
