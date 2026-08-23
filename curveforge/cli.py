from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import GeneratorConfig
from .generator import DatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="curveforge",description="Generate plot images and exact curve masks")
    p.add_argument("--output","-o",default="dataset",help="Output directory")
    p.add_argument("--count","-n",type=int,default=1000,help="Number of images")
    p.add_argument("--config",type=Path,help="JSON configuration")
    p.add_argument("--seed",type=int,help="Override random seed")
    p.add_argument("--width",type=int,help="Override image width")
    p.add_argument("--height",type=int,help="Override image height")
    p.add_argument("--plot-domain",choices=("general","dsc","mixed"),help="Override plot domain")
    p.add_argument("--val-fraction",type=float,default=.1)
    p.add_argument("--test-fraction",type=float,default=.1)
    p.add_argument("--workers","-j",type=int,default=1,help="Parallel worker processes")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted generation and reuse complete samples",
    )
    p.add_argument("--write-default-config",type=Path,help="Write default JSON config and exit")
    return p


def main(argv: list[str]|None=None) -> int:
    args=build_parser().parse_args(argv)
    if args.write_default_config:
        args.write_default_config.write_text(json.dumps(GeneratorConfig().to_dict(),indent=2),encoding="utf-8")
        print(f"Wrote {args.write_default_config}")
        return 0
    cfg=GeneratorConfig.from_json(args.config) if args.config else GeneratorConfig()
    for key in ("seed","width","height","plot_domain"):
        value=getattr(args,key)
        if value is not None: setattr(cfg,key,value)
    cfg.validate()
    result=DatasetGenerator(cfg).generate(
        args.output,
        args.count,
        args.val_fraction,
        args.test_fraction,
        args.workers,
        resume=args.resume,
    )
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0
