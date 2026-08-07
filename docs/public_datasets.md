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

## Balanced LineEX v5 training set

The production recipe takes a deterministic 100,000-image subset of the
official LineEX train split and keeps its complete 10,000-image validation and
20,000-image test splits. CurveForge mirrors those split sizes with the calmer
`configs/balanced_lineex_v5.json` profile. Training therefore contains exactly
100,000 public and 100,000 CurveForge images (50/50).

The cap is intentional: all 400,000 LineEX training images plus an equal
synthetic half would quadruple MaskDINO training time. Official validation and
test splits remain untouched for comparable evaluation.

```bash
bash server/submit_balanced_lineex_v5.sh
```

The three resumable jobs download and convert LineEX, generate matching
synthetic splits, merge with hard links, and write compressed COCO RLE.
