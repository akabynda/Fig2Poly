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
refinement. Both use exclusive horizontal and vertical flips with probabilities
0.3 and 0.3. PMC and Adobe additionally use the original random shifts (probability
0.3, offsets from -51 through 50 pixels) and absolute 435-pixel crops (probability
0.3), before fixed resize/pad. LineEX and DSC use the original LineEX branch without
those extra transforms. MaskDINO recomputes boxes and removes empty transformed
instances because it has box losses; Mask2Former follows the original mask pipeline.

## Loss functions and weights

Verified on 2026-09-09 against the saved full Mask2Former configuration, the saved
MaskDINO R50 smoke configuration, and the deployed loss implementations. The
MaskDINO full-training profile uses the same loss settings as the smoke run;
the smoke run overrides iteration, evaluation, checkpoint and logging intervals.

| Component | Mask2Former Swin-Tiny / LineFormer | MaskDINO R50 |
| --- | --- | --- |
| Classification | Softmax cross-entropy, `loss_cls`, weight **1** | Sigmoid focal loss, `loss_ce`, weight **4**, alpha **0.25**, gamma **2** |
| Mask pixels | Binary cross-entropy with logits, `loss_mask`, weight **5** | Binary cross-entropy with logits, `loss_mask`, weight **5** |
| Mask overlap | Sigmoid Dice, `loss_dice`, weight **5**, naive Dice with smoothing **1** | Sigmoid Dice, `loss_dice`, weight **5**, smoothing **1** |
| Box coordinates | No box loss | L1 on normalized center-x/center-y/width/height, `loss_bbox`, weight **5** |
| Box overlap | No box loss | `1 - GIoU`, `loss_giou`, weight **2** |

Thus the final prediction's weighted loss is:

```text
Mask2Former: 1 * CE_class + 5 * BCE_mask + 5 * Dice_mask
MaskDINO:    4 * Focal_class + 5 * BCE_mask + 5 * Dice_mask
             + 5 * L1_box + 2 * GIoU_box
```

The total training loss also sums auxiliary prediction losses:

- Mask2Former supervises the initial prediction and all nine decoder layers.
  The final layer uses `loss_cls`, `loss_mask`, `loss_dice`; the preceding nine
  prediction sets use `d0.*` through `d8.*`, with the same component weights.
  Classification class weights are `[1.0, 0.1]` for `line` and `no-object`.
- MaskDINO has `DEEP_SUPERVISION=True`, `DEC_LAYERS=9`, `INITIAL_PRED=True`,
  `TWO_STAGE=True`. Auxiliary decoder predictions and intermediate two-stage
  predictions receive the corresponding component weights, without an additional
  discount. Denoising is enabled as `DN=seg`, `DN_NUM=100`, `DN_NOISE_SCALE=0.4`;
  its classification, mask and box losses also use the weights above. The saved
  `NO_OBJECT_WEIGHT=0.1` belongs to the alternative softmax CE branch; the active
  instance-classification branch uses focal loss (`SEMANTIC_CE_LOSS=False`).

Both models sample **12,544 mask points**, with oversampling ratio **3.0** and
importance sampling ratio **0.75**. MaskDINO's `loss_mask` is BCE, even though an
upstream method docstring calls it focal loss. Focal loss is used for classification.
The custom centerline/tangent curve loss is disabled (`curve_loss_weight=0`).

Hungarian matching costs are separate from the optimized loss weights:

| Matching cost | Mask2Former | MaskDINO |
| --- | --- | --- |
| Class / mask BCE / Dice | **2 / 5 / 5** | **4 / 5 / 5** |
| Box L1 / GIoU | Not used | **5 / 2** |

Implementation sources are `third_party/LineFormer/lineformer_swin_t_config.py`,
the vendored MMDetection `mask2former_head.py` and `maskformer_head.py`, and
`third_party/MaskDINO/maskdino/maskdino.py` plus `modeling/criterion.py` on the server.
The training wrappers preserve these losses. Their shared LineFormer profile
sets the training schedule, data pipeline and query count.

## Saved configurations

The server run root is `/mnt/tank/scratch/devdokimov/Fig2Poly_runs`.

| Run subdirectory | Saved files |
| --- | --- |
| `mask2former_swin_t_lineformer_originals_dsc_exact_100k_v1` | `lineformer_dsc_finetune_config.py`: resolved model, losses, optimizer, schedule and data pipeline; `lineformer_recipe_request.json` and `finetune_request.json`: launch arguments |
| `maskdino_r50_lineformer_originals_dsc_exact_100k_v1` | `config.yaml`: resolved Detectron2/MaskDINO configuration; `training_profile.json`: schedule and overrides; written when this queued training starts |
| `lineformer_originals_dsc_exact_smoke_v1/11713/maskdino` | Existing `config.yaml` and `training_profile.json` from the successful smoke run; loss settings match the full profile, but its schedule is only two iterations |

Full runs also save `mixture_summary.json` and `fig2poly_git_sha.txt`. The historical
filename `lineformer_dsc_finetune_config.py` is shared with the older wrapper; its
name does not imply fine-tuning from the old DSC checkpoint. This experiment uses
ImageNet backbone initialization and a new segmentation head.

## Data and masks

- PMC: official training release with deterministic 10% validation holdout;
  remaining training images repeated 50 times, matching the original weighting.
- AdobeSynth: all eligible line/scatter-line charts from the official training
  release, with a deterministic 10% validation holdout; train repetition 1.
  The completed ten-shard extraction contains 41,874 eligible images and seven
  orphan line annotations: `162724`, `168329`, `176337`, `182570`, `187288`,
  `192989`, `196163`. Only these missing pairs may be excluded; any other missing
  image fails preparation. Available images with these IDs are always included.
  `adobe_line_image_audit.json` records actual exclusions and the download manifest
  records archive sizes, SHA256 hashes and extraction counts. The converter also
  records exclusions in the Adobe training receipt and COCO metadata.
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
