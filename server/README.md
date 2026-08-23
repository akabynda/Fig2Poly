# Server training

This directory prepares three prompt-free instance-segmentation experiments on
exactly the same train/validation/test split:

- Ultralytics `YOLO26x-seg`;
- official MaskDINO with ResNet-50;
- official MaskDINO with Swin-L.

## 1. Configure and inspect the server

```bash
cp server/.env.example server/.env
nano server/.env
bash server/check_server.sh
```

Keep `DATA_ROOT`, `RUNS_ROOT`, and `CACHE_ROOT` on the large data disk. Do not
commit `server/.env`. Set `TORCH_CUDA_ARCH_LIST=8.0` for A100 or `9.0` for
H100 before building MaskDINO. H100 also requires a modern CUDA toolkit and
PyTorch build.

## 2. Install isolated environments

```bash
bash server/setup_yolo.sh
bash server/setup_maskdino.sh
bash server/setup_lineformer.sh
```

MaskDINO contains compiled CUDA deformable-attention operators. The setup must
be run on the final server after its driver, CUDA toolkit, and GPU are known.

## 3. Build the data

```bash
bash server/generate_synthetic.sh
bash server/download_public.sh
bash server/convert_public.sh
bash server/prepare_training_data.sh
```

The current balanced production recipe is submitted as a resumable dependency
chain:

```bash
bash server/submit_balanced_lineex_v5.sh
```

It uses 100k LineEX train images plus 100k calmer CurveForge images. Complete
official LineEX validation/test splits are mirrored by synthetic splits and
kept separate from training.

Generation and downloads are resumable. `combined` uses hard links when source
and destination are on the same filesystem. Both YOLO and MaskDINO are prepared
from the same combined JSONL manifests.

MaskDINO receives exact compressed COCO RLE masks, including disconnected
dashes and visible-only gaps. Native YOLO labels are polygons and cannot encode
one disconnected instance exactly; its converter joins components and records
this limitation in `yolo_conversion.json`.

The v5 MaskDINO job keeps BCE + Dice instance supervision and adds a
thickness-invariant centreline/tangent auxiliary loss. Validation reports the
LineFormer/ChartInfo 6a and 6b continuous-line scores; early stopping follows
the count-penalized 6b score instead of COCO AP alone.

### DSC fine-tuning for LineFormer

Point `DSC_DATASET` in `server/.env` at a completed CurveForge dataset, then
prepare LineFormer's COCO input and start fine-tuning:

```bash
bash server/prepare_lineformer_dsc.sh
bash server/train_lineformer_dsc.sh
```

On Slurm:

```bash
prep_job=$(sbatch --parsable server/slurm/prepare_lineformer_dsc.sbatch)
sbatch --dependency="afterok:$prep_job" server/slurm/train_lineformer_dsc.sbatch
```

The converter names the single category `line`, dilates train targets to 3 px,
and leaves validation/test masks exact. This makes one-pixel DSC traces stable
after LineFormer's 512 px resize without inflating evaluation scores. Empty
CurveForge plots remain in the training set as hard negatives. Training starts
from `LINEFORMER_WEIGHTS` and automatically resumes from `latest.pth`.

Defaults are conservative for fine-tuning: 10,000 iterations, learning rate
`2e-5`, two samples per GPU, validation/checkpointing every 500 iterations.
Use the untouched test split only after selecting the checkpoint and confidence
threshold on validation.

## 4. Smoke test before full training

Set a small `SYNTH_COUNT` and low `EPOCHS` in `.env`, then run each script once.
Include empty charts in the smoke data. Official MaskDINO denoising was not
designed specifically for empty targets, so a successful empty-sample step is a
required gate before the full run.

## 5. Train and resume

Run each command in its own `tmux` session:

```bash
bash server/train_yolo26x_seg.sh
bash server/train_maskdino_r50.sh
bash server/train_maskdino_swinl.sh
```

The scripts resume automatically when `last.pt` or Detectron2
`last_checkpoint` exists. YOLO checkpoints are epoch-boundary checkpoints;
MaskDINO checkpoints are written once per computed epoch. Keep the GPU count
and global batch unchanged when resuming.

Swin-L used roughly 60 GB per GPU at two images per GPU in the official setup.
Start with global batch 1 on a single 80 GB GPU. R50 is the safer first run.

## 6. Evaluation

```bash
bash server/evaluate_all.sh
bash server/status.sh
```

The script evaluates the untouched test split with each framework's COCO mask
metrics. Threshold selection and any custom centerline metric must be performed
on validation first and then frozen for test.
