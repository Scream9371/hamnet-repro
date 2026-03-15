# Evaluation Protocols

This repository keeps only the two fixed evaluation protocols used by the paper.

## Protocols

- `samplewise`: fixed sample-wise split.
- `cpdp`: fixed cross-project split.

## Split Directories

```text
splits/
├── cpdp/
└── samplewise/
```

## Design Principle

Protocol differences should be controlled by the split files and protocol selection in the runner, not by introducing multiple unrelated model configurations.
