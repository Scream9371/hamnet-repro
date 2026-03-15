# Dataset Notes

This repository does not directly ship the dataset files.

## Source

- <https://huggingface.co/datasets/Scream9371/hamnet-datasets>

## Target Location

Place the downloaded JSONL files under the repository-level `datasets/` directory.

## Expected File Names

- `promise_java.jsonl`
- `defactors_python.jsonl`
- `bigvul_c.jsonl`
- `bugsinpy_python.jsonl`
- `devign_c.jsonl`

## Recommended Download Command

```bash
huggingface-cli download Scream9371/hamnet-datasets \
  --repo-type dataset \
  --local-dir datasets
```

## Notes

- Keep the file names unchanged so that the fixed paths used by the runners remain valid.
- If an automated download helper is added later, it should still target `datasets/`.
