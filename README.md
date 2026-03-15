# hamnet-repro

This repository is a compact reproducibility package for the main experiments of HAM-Net. It is organized around the fixed protocols, configurations, and reference summaries that directly support the results reported in the paper.

## Quick Start

1. Create and activate the environment:

```bash
conda env create -f environment/dl_env.yml
conda activate ham-net
```

2. Download the datasets into `datasets/` from Hugging Face:

- <https://huggingface.co/datasets/Scream9371/hamnet-datasets>

Example:

```bash
hf download Scream9371/hamnet-datasets \
  --repo-type dataset \
  --local-dir datasets
```

Recommended one-time local setup for reproducibility:

1. Download the datasets:

```bash
hf download Scream9371/hamnet-datasets \
  --repo-type dataset \
  --local-dir datasets
```

2. Download CodeBERT to a local checkpoint directory:

```bash
hf download microsoft/codebert-base \
  --local-dir checkpoints/codebert-base
```

3. Update the config files to use the local checkpoint:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(".")
for rel in ["config/main.json", "config/ablation.json"]:
    path = root / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    data["common"]["encoder_local_path"] = "checkpoints/codebert-base"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
```

After these three steps, the repository is ready to run without downloading CodeBERT during execution.

After download, the repository should contain:

```text
datasets/
├── promise_java.jsonl
├── defactors_python.jsonl
├── bigvul_c.jsonl
├── bugsinpy_python.jsonl
└── devign_c.jsonl
```

3. Review the protocol notes and result mapping:

- [docs/quickstart.md](/home/scream9371/hamnet-repro/docs/quickstart.md)
- [docs/evaluation_protocols.md](/home/scream9371/hamnet-repro/docs/evaluation_protocols.md)
- [docs/results_map.md](/home/scream9371/hamnet-repro/docs/results_map.md)

## CodeBERT Checkpoint

- Recommended setup for this repository: download CodeBERT once to `checkpoints/codebert-base` and set `encoder_local_path` to that directory in `config/main.json` and `config/ablation.json`.
- This is the most review-friendly option because it avoids runtime dependence on external network access and makes the encoder location explicit.
- If `encoder_local_path` is left as `null`, the scripts fall back to `microsoft/codebert-base`.
- In that fallback mode, `transformers.from_pretrained(...)` downloads the checkpoint from Hugging Face only when it is missing from the local cache. Later runs reuse the cache and do not download the encoder again.

Example local download:

```bash
hf download microsoft/codebert-base \
  --local-dir checkpoints/codebert-base
```

Then set:

```json
"encoder_name": "microsoft/codebert-base",
"encoder_local_path": "checkpoints/codebert-base"
```

Recommended interpretation:

- Prefer `checkpoints/codebert-base` for artifact evaluation and offline execution.
- Use `microsoft/codebert-base` as a fallback only when online access is acceptable.

## Repository Layout

- `datasets/`: dataset files downloaded by the user.
- `splits/`: fixed split files for the paper protocols.
- `caps/`: deterministic function subset files used during training.
- `config/`: main and ablation configurations.
- `src/`: minimal retained code for HAM-Net, baselines, and runnable entry points.
- `results_reference/`: reference summaries aligned with the reported paper results.

## Notes

- This repository focuses on reproducibility of the paper’s primary experiments.
- All paths are organized relative to the repository root for portability across machines.
