# paravt.data

After you `huggingface-cli download ParaVT/ParaVT-Source` and extract the per-bucket zip archives, the parquets in [`ParaVT/ParaVT-Parquet`](https://huggingface.co/datasets/ParaVT/ParaVT-Parquet) still hold *sentinel-relative* video paths (`longvt_source/<src>/<rel>`, `museg/<sub>/<id>.mp4`, `selfqa/<id>.mp4`). This module prepends your local extraction root to every sentinel path so downstream training/eval can consume `file://` URIs directly.

## Sentinel scheme

The release uses four virtual roots to keep the parquets cluster-independent:

| Sentinel | Source zips in ParaVT-Source |
|---|---|
| `longvt_source/<src>/<rel>` | re-zipped per LongVT source dir (`videor1_*`, `longvideoreason_*`, ...) |
| `museg/charades/<id>.mp4` | `museg/charades/*.zip` |
| `museg/et_instruct_164k/<id>.mp4` | `museg/et_instruct_164k/*.zip` |
| `selfqa/<id>.mp4` | `selfqa/*.zip` |

`materialize` simply prefixes each sentinel with your extraction root and re-attaches the `file://` scheme.

## Typical user flow

```bash
# 1. parquets — small (~200 MB)
huggingface-cli download ParaVT/ParaVT-Parquet --repo-type dataset --local-dir ./paravt-parquet

# 2. videos + auxiliary images — large; pick the splits you need.
#    Each archive's members carry the full sentinel path, so extract every
#    zip into the same top-level root.
huggingface-cli download ParaVT/ParaVT-Source --repo-type dataset --local-dir ./paravt-source
( cd ./paravt-source && find . -name "*.zip" -exec unzip -q -o -d . {} \; )

# 3. materialize the paths once
python -m paravt.data.materialize \
    --root        ./paravt-source \
    --parquet-dir ./paravt-parquet \
    --output-dir  ./paravt-parquet-materialized
```

Materialized parquets are drop-in for `lmms-engine` supervised fine-tuning and the `paravt.eval` driver. See [`paravt/sft/README.md`](../sft/README.md) and [`paravt/eval/README.md`](../eval/README.md) for the next step.
