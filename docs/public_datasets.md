# Storage-efficient public datasets

Download and extract the official datasets:

```powershell
python -m training.download_public_benchmarks `
  --root datasets/public `
  --extract `
  --delete-archives
```

For Adobe CHART-Synthetic, annotations are processed first. The large image
archives then extract only eligible `line` and `scatter line` charts that have
Task 6 curves. macOS metadata files and all unrelated chart types are skipped.

Per-asset receipts in `datasets/public/_state` make interrupted downloads
resumable without fetching completed archives again. A filtered Adobe archive
is marked complete only after the downloader verifies that every eligible
annotation has a corresponding extracted image.

After successful normalization, the P2 pipeline verifies every normalized
image and curve mask before deleting the raw Adobe files:

```powershell
python -m training.run_p2_pipeline
```

Use `--no-prune-adobe-raw` to retain the Adobe sources for debugging. Raw
LineEX and UB-PMC sources are not removed by this cleanup policy.
