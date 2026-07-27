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

The generic conversion utility writes independent instance masks and JSONL
manifests. These files can later be adapted to the dataset format required by
the selected training framework:

```powershell
python -m training.convert_public_instances --help
```

Raw public datasets are not removed automatically by the converter.
