# Legacy Course Snapshots

This directory preserves the original COMP5046 assignment material that was merged into this repo.

## What is here

- `A2/`: the perceptron-style Q6 code model assignment snapshot.
- `A4/`: the ATIS-based Text2SQL assignment snapshot, including preprocessing, classification, generation, LLM prompts, processed splits, and historical checkpoints.

## Why it looks duplicated

The source workspace contained both a flattened A4 root and a nested `A4/` copy. Both were preserved so the original file layout stays visible and no script path is silently dropped.

## Read order

1. [`A2/README.md`](./A2/README.md)
2. [`A4/README.md`](./A4/README.md)

## Notes

- Large historical artifacts such as `.pt` checkpoints and `Archive.zip` are intentionally kept here for traceability.
- The OpenRouter-based scripts read `OPENROUTER_API_KEY` from the environment after the cleanup in this repo.
