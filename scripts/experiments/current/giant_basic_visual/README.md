# Giant Basic Visual

Shadow experiment for testing whether the Basic5 giant model improves when it
gets features from the main image only.

Rules:

- Only `LocalPrimaryImagePath` is used.
- Rows with missing or unreadable main images are skipped.
- `LocalImagePaths` is never used as fallback.
- Telegram policy is not changed by this package.
- Live scoring is shadow-only.

Outputs stay under:

```text
data/experiments/giant_basic_visual/
```

