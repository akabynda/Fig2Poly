# LineFormer schedule with public data and exact DSC masks

This experiment trains Mask2Former Swin-Tiny (the original LineFormer architecture)
and MaskDINO R50 using one prepared COCO mixture. Each training uses two Slurm GPUs;
MaskDINO starts after Mask2Former releases its allocation. Both require a successful
two-GPU smoke test covering training, evaluation and checkpoint writing.

| Setting | Both full trainings |
| --- | --- |
| Runner | Iteration based, 100,000 optimizer updates |
| Batch | 4 images per GPU, 8 total, no accumulation or LR autoscaling |
| Optimizer | AdamW, LR 0.0001, weight decay 0.05 |
| LR schedule | Gamma 0.75 every 5,000 updates; warmup factor 1 for 10 updates |
| Gradient clipping | Global L2 norm 0.01 |
| Validation / checkpoints / logs | Every 250 / 500 / 100 updates |
| Best checkpoint | Validation segmentation AP; no early stopping |
| Periodic checkpoint retention | Three, plus best and final |
| Training image | Fit into 512 by 512, white padding |
| Queries / classes | 100 / one `line` class |
| Precision / seed | FP32 / 20260905 |

Initialization uses an ImageNet Swin-Tiny or ImageNet ResNet-50 backbone and a new
segmentation head. The previously trained LineFormer, DSC and COCO segmentation
checkpoints are not used. MaskDINO retains its own losses, denoising and box
refinement. Both use the LineEX augmentation branch: exclusive horizontal and
vertical flips with probabilities 0.3 and 0.3, then fixed resize/pad. The original
PMC/Adobe-specific shift/crop branch is not used.

## Data and masks

- PMC: official training release with deterministic 10% validation holdout;
  remaining training images repeated 50 times, matching the original weighting.
- AdobeSynth: all eligible line/scatter-line charts from the official training
  release, with a deterministic 10% validation holdout; train repetition 1.
- LineEX: all available official training images (400,000 released), with its
  official validation and test splits; train repetition 1.
- DSC: the existing `coco_lineformer_dsc_exact` corresponding to `dataset_dsc`
  used by `lineformer_dsc_5ep_earlystop`: 80,000 train, 10,000 validation, 10,000 test.
  Its existing masks are preserved without dilation.

Public annotations provided as polylines are rasterized at 1 pixel with annotated
occlusions removed. They are proxy masks reconstructed from official annotations.
The authors' exact COCO exports, subset IDs and original split boundaries are not
available on this server. The source families and training settings are reproduced;
the exact historical LineFormer dataset is not claimed to be reproduced.

All four validation sources are evaluated every 250 updates; official test images
are kept out of checkpoint selection. Source fingerprints, counts, masks and
sampling weights are recorded in `preparation_manifest.json`, `recipe.json` and
`mixture_summary.json`. Train images are checked for overlap with validation/test.

## Server launch

```bash
bash server/submit_lineformer_originals_dsc.sh
```

Submission receipts live under `Fig2Poly_runs/lineformer_originals_dsc_exact_100k_v1`.
Repeating the command reuses those receipts instead of submitting duplicate jobs.
CPU preparation downloads the missing Adobe data to a new directory and builds
`Fig2Poly_data/coco_lineformer_originals_dsc_exact_v1`. Existing data are reused
without modification. Large COCO outputs are streamed and images are hardlinked
where possible.

Full outputs:

- `Fig2Poly_runs/mask2former_swin_t_lineformer_originals_dsc_exact_100k_v1`
- `Fig2Poly_runs/maskdino_r50_lineformer_originals_dsc_exact_100k_v1`

The jobs request QoS `devdokimov` and `gpu:2`. Slurm time-limit signals stop the
trainer process group and requeue from the latest checkpoint. Resumption restores
optimizer/scheduler progress, but does not promise bitwise identical augmentation
randomness across interruptions. A full validation every 250 steps can contribute
substantial runtime on this combined dataset.
