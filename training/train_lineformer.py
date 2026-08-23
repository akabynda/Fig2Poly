from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def dataset_config(annotation: Path,image_dir: Path,pipeline: list,category_name: str="line") -> dict:
    return {
        "type":"CocoDataset",
        "classes":(category_name,),
        "ann_file":str(annotation),
        "img_prefix":str(image_dir)+os.sep,
        "pipeline":pipeline,
        # CurveForge intentionally contains empty plots as hard negatives.
        "filter_empty_gt":False,
    }


def validate_dataset(dataset: Path) -> None:
    missing=[]
    for split in ("train","val","test"):
        for path in (
            dataset/"annotations"/f"instances_{split}.json",
            dataset/"images"/split,
        ):
            if not path.exists(): missing.append(str(path))
    if missing:
        raise FileNotFoundError("LineFormer COCO dataset is incomplete:\n"+"\n".join(missing))
    for split in ("train","val","test"):
        payload=json.loads((dataset/"annotations"/f"instances_{split}.json").read_text(encoding="utf-8"))
        categories={item["name"] for item in payload.get("categories",[])}
        if categories!={"line"}:
            raise ValueError(f"{split} categories must be {{'line'}}, got {categories}")


def build_config(args: argparse.Namespace) -> Path:
    root=args.lineformer_root.resolve(); dataset=args.dataset.resolve(); output=args.output.resolve()
    config_path=root/"lineformer_swin_t_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"LineFormer config not found: {config_path}")
    validate_dataset(dataset)
    sys.path[:0]=[str(root/"mmdetection"),str(root)]
    from mmcv import Config

    cfg=Config.fromfile(str(config_path))
    train_pipeline=cfg.train_pipeline_LineEX
    test_pipeline=cfg.test_pipeline
    cfg.data={
        "samples_per_gpu":args.samples_per_gpu,
        "workers_per_gpu":args.workers_per_gpu,
        "train":dataset_config(
            dataset/"annotations"/"instances_train.json",dataset/"images"/"train",
            train_pipeline,
        ),
        "val":dataset_config(
            dataset/"annotations"/"instances_val.json",dataset/"images"/"val",
            test_pipeline,
        ),
        "test":dataset_config(
            dataset/"annotations"/"instances_test.json",dataset/"images"/"test",
            test_pipeline,
        ),
    }
    cfg.work_dir=str(output)
    cfg.load_from=str(args.weights.resolve()) if args.weights else None
    cfg.resume_from=None
    cfg.runner={"type":"IterBasedRunner","max_iters":args.max_iters}
    cfg.max_iters=args.max_iters
    cfg.workflow=[("train",1)]
    cfg.optimizer.lr=args.base_lr
    cfg.evaluation={"interval":args.eval_interval,"metric":["segm"],"save_best":"segm_mAP"}
    cfg.checkpoint_config={
        "interval":args.checkpoint_interval,"by_epoch":False,"save_last":True,"max_keep_ckpts":3,
    }
    cfg.log_config.interval=args.log_interval
    cfg.seed=args.seed
    cfg.gpu_ids=list(range(args.num_gpus))
    cfg.auto_scale_lr={"enable":False,"base_batch_size":16}
    output.mkdir(parents=True,exist_ok=True)
    generated=output/"lineformer_dsc_finetune_config.py"
    cfg.dump(str(generated))
    summary={
        "lineformer_root":str(root),"dataset":str(dataset),"weights":str(args.weights.resolve()) if args.weights else None,
        "max_iters":args.max_iters,"base_lr":args.base_lr,"samples_per_gpu":args.samples_per_gpu,
        "num_gpus":args.num_gpus,"train_mask_note":"COCO train masks may be dilated; val/test masks are exact",
    }
    (output/"finetune_request.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return generated


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description="Fine-tune LineFormer on CurveForge COCO masks")
    parser.add_argument("--lineformer-root",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",type=Path,help="Pretrained LineFormer checkpoint")
    parser.add_argument("--max-iters",type=int,default=10000)
    parser.add_argument("--base-lr",type=float,default=2e-5)
    parser.add_argument("--samples-per-gpu",type=int,default=2)
    parser.add_argument("--workers-per-gpu",type=int,default=4)
    parser.add_argument("--num-gpus",type=int,default=1)
    parser.add_argument("--eval-interval",type=int,default=500)
    parser.add_argument("--checkpoint-interval",type=int,default=500)
    parser.add_argument("--log-interval",type=int,default=50)
    parser.add_argument("--seed",type=int,default=20260903)
    parser.add_argument("--resume",action="store_true",help="Auto-resume from output/latest.pth")
    parser.add_argument("--dry-run",action="store_true",help="Write config without starting training")
    args=parser.parse_args(argv)
    for name in ("max_iters","samples_per_gpu","workers_per_gpu","num_gpus",
                 "eval_interval","checkpoint_interval","log_interval"):
        if getattr(args,name)<1: parser.error(f"--{name.replace('_','-')} must be positive")
    if args.base_lr<=0: parser.error("--base-lr must be positive")
    if args.weights and not args.weights.is_file(): parser.error(f"weights not found: {args.weights}")
    generated=build_config(args)
    if args.dry_run:
        print(generated)
        return 0
    root=args.lineformer_root.resolve(); train_py=root/"mmdetection"/"tools"/"train.py"
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(
        [str(root/"mmdetection"),str(root),env.get("PYTHONPATH","")]
    ).rstrip(os.pathsep)
    common=[str(train_py),str(generated),"--work-dir",str(args.output.resolve()),
            "--seed",str(args.seed)]
    if args.resume: common.append("--auto-resume")
    if args.num_gpus==1:
        command=[sys.executable,*common,"--gpu-id","0"]
    else:
        command=[sys.executable,"-m","torch.distributed.launch","--nproc_per_node",str(args.num_gpus),
                 *common,"--launcher","pytorch"]
    print("Running:"," ".join(command),flush=True)
    return subprocess.run(command,env=env,check=False).returncode


if __name__=="__main__":
    raise SystemExit(main())
