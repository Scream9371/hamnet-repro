# Quick Start

## 1. Environment

```bash
conda env create -f environment/dl_env.yml
conda activate ham-net
```

## 2. Dataset Download

Download the dataset files from:

- <https://huggingface.co/datasets/Scream9371/hamnet-datasets>

Recommended command:

```bash
hf download Scream9371/hamnet-datasets \
  --repo-type dataset \
  --local-dir datasets
```

## 3. Available Fixed Protocols

- `splits/cpdp/`: fixed cross-project protocol.
- `splits/samplewise/`: fixed sample-wise protocol.

## 4. Configuration Files

- `config/main.json`: main experiment configuration.
- `config/ablation.json`: ablation variants used in the paper.

## 5. CodeBERT Download Behavior

- Recommended setup: pre-download CodeBERT to `checkpoints/codebert-base` and set `encoder_local_path` in the config files.
- This makes the repository easier to run in offline, mirrored, or restricted review environments.
- If `encoder_local_path` is `null`, HAM-Net falls back to `microsoft/codebert-base`.
- In that fallback mode, `transformers` downloads the checkpoint only on the first run when the local cache is missing.
- Later runs reuse the local Hugging Face cache instead of downloading the checkpoint again.

Example:

```bash
hf download microsoft/codebert-base \
  --local-dir checkpoints/codebert-base
```

Then update `config/main.json` and `config/ablation.json`:

```json
"encoder_name": "microsoft/codebert-base",
"encoder_local_path": "checkpoints/codebert-base"
```

Practical recommendation:

- Use the local checkpoint path above for reproducibility-focused runs.
- Leave `encoder_local_path` as `null` only if network access to Hugging Face is expected during the first run.

## 6. Reference Result Files

- `results_reference/samplewise_summary.json`
- `results_reference/cpdp_summary.json`
- `results_reference/ablation_summary.json`
- `results_reference/interpretability_summary.json`

These files are intended to store the final reference summaries aligned with the paper tables and figures.
