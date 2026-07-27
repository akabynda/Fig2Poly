from pathlib import Path
import json
import random

import numpy as np
from PIL import Image

from curveforge.config import GeneratorConfig
from curveforge.generator import DatasetGenerator


def test_tiny_dataset(tmp_path: Path):
    cfg=GeneratorConfig(width=192,height=160,supersample=1,max_curves=3,max_points=150,seed=7)
    summary=DatasetGenerator(cfg).generate(tmp_path,3,val_fraction=1/3,test_fraction=1/3)
    assert summary["splits"]=={"train":1,"val":1,"test":1}
    for split,index in (("train",0),("val",1),("test",2)):
        image=Image.open(tmp_path/"images"/split/f"{index:08d}.jpg")
        semantic=np.asarray(Image.open(tmp_path/"semantic_masks"/split/f"{index:08d}.png"))
        instance=np.asarray(Image.open(tmp_path/"instance_masks"/split/f"{index:08d}.png"))
        assert image.size==(192,160)
        assert semantic.shape==instance.shape==(160,192)
        assert np.array_equal(semantic>0,instance>0)
        curve_masks=sorted((tmp_path/"curve_masks"/split/f"{index:08d}").glob("curve_*.png"))
        metadata=json.loads((tmp_path/"metadata"/split/f"{index:08d}.json").read_text())
        assert len(curve_masks)==metadata["curve_count"]==len(metadata["curves"])
        union=np.zeros_like(semantic,dtype=bool)
        for mask_path,curve in zip(curve_masks,metadata["curves"]):
            curve_mask=np.asarray(Image.open(mask_path))>0
            assert curve_mask.any()
            assert curve["mask"].endswith(mask_path.name)
            union |= curve_mask
        assert np.array_equal(union,semantic>0)


def test_reproducible(tmp_path: Path):
    cfg=GeneratorConfig(width=128,height=128,supersample=1,max_curves=2,max_points=100,seed=9)
    DatasetGenerator(cfg).generate(tmp_path/"a",1,0,0)
    DatasetGenerator(cfg).generate(tmp_path/"b",1,0,0)
    a=(tmp_path/"a"/"semantic_masks"/"train"/"00000000.png").read_bytes()
    b=(tmp_path/"b"/"semantic_masks"/"train"/"00000000.png").read_bytes()
    assert a==b


def test_interrupted_generation_can_resume(tmp_path: Path):
    cfg=GeneratorConfig(width=128,height=128,supersample=1,max_curves=2,max_points=100,seed=11)
    generator=DatasetGenerator(cfg)
    first=generator.generate(tmp_path,3,val_fraction=1/3,test_fraction=1/3)
    assert first["generated_now"]==3
    resumed=generator.generate(tmp_path,3,val_fraction=1/3,test_fraction=1/3,resume=True)
    assert resumed["generated_now"]==0
    assert resumed["reused_existing"]==3
    state=json.loads((tmp_path/"generation_state.json").read_text(encoding="utf-8"))
    assert state["status"]=="completed"


def test_resume_rejects_incompatible_settings(tmp_path: Path):
    cfg=GeneratorConfig(width=128,height=128,supersample=1,max_curves=2,max_points=100,seed=13)
    DatasetGenerator(cfg).generate(tmp_path,1,0,0)
    incompatible=GeneratorConfig(width=160,height=128,supersample=1,max_curves=2,max_points=100,seed=13)
    try:
        DatasetGenerator(incompatible).generate(tmp_path,1,0,0,resume=True)
    except ValueError as error:
        assert "incompatible settings" in str(error)
    else:
        raise AssertionError("incompatible resume must fail")


def test_opaque_legend_is_removed_from_curve_masks():
    cfg=GeneratorConfig(
        width=320,
        height=240,
        supersample=1,
        min_curves=3,
        max_curves=3,
        max_points=300,
        legend_probability=1,
        title_probability=0,
        labels_probability=0,
        annotations_probability=0,
        occlusion_probability=0,
        multi_panel_probability=0,
    )
    generator=DatasetGenerator(cfg)
    found_opaque_legend=False
    for seed in range(20):
        rng=random.Random(seed)
        image,masks,metadata=generator._render_base(rng,np.random.default_rng(seed))
        legend=next(item for item in metadata["occluders"] if item["type"]=="legend")
        if not legend["opaque_background"]:
            continue
        found_opaque_legend=True
        x0,y0,x1,y1=legend["bbox"]
        for mask in masks:
            assert not (np.asarray(mask)[y0:y1+1,x0:x1+1]>0).any()
        assert image.size==(320,240)
        break
    assert found_opaque_legend


def test_empty_plot_is_a_valid_negative_example(tmp_path: Path):
    cfg=GeneratorConfig(
        width=160,
        height=128,
        supersample=1,
        empty_plot_probability=1,
        page_layout_probability=1,
        hard_negatives_probability=1,
        multi_panel_probability=0,
        max_points=100,
    )
    DatasetGenerator(cfg).generate(tmp_path,1,0,0)
    metadata=json.loads((tmp_path/"metadata"/"train"/"00000000.json").read_text())
    semantic=np.asarray(Image.open(tmp_path/"semantic_masks"/"train"/"00000000.png"))
    assert metadata["curve_count"]==metadata["generated_curve_count"]==0
    assert metadata["page_layout"]
    assert metadata["hard_negative_count"]>0
    assert not semantic.any()
    assert not list((tmp_path/"curve_masks"/"train"/"00000000").glob("*.png"))


def test_non_polynomial_families_are_recorded():
    cfg=GeneratorConfig(
        width=160,
        height=128,
        supersample=1,
        min_curves=5,
        max_curves=5,
        empty_plot_probability=0,
        non_polynomial_probability=1,
        multi_panel_probability=0,
        max_points=120,
    )
    generator=DatasetGenerator(cfg)
    _,masks,metadata=generator._render_base(random.Random(123),np.random.default_rng(123))
    assert len(masks)==5
    assert all(curve["family"]!="polynomial" for curve in metadata["curves"])
    assert all(curve["function_parameters"] for curve in metadata["curves"])


def test_related_curve_group_is_recorded():
    cfg=GeneratorConfig(
        width=160,
        height=128,
        supersample=1,
        min_curves=4,
        max_curves=4,
        empty_plot_probability=0,
        related_curves_probability=1,
        multi_panel_probability=0,
        max_points=120,
    )
    generator=DatasetGenerator(cfg)
    _,_,metadata=generator._render_base(random.Random(456),np.random.default_rng(456))
    related=[curve for curve in metadata["curves"] if curve["relation_group"]==1]
    assert 2<=len(related)<=4
    assert len({curve["family"] for curve in related})==1


def test_multi_panel_keeps_one_mask_per_curve():
    cfg=GeneratorConfig(
        width=320,
        height=240,
        supersample=1,
        min_curves=2,
        max_curves=2,
        empty_plot_probability=0,
        multi_panel_probability=1,
        page_layout_probability=0,
        max_points=120,
    )
    generator=DatasetGenerator(cfg)
    _,masks,metadata=generator._render_base(random.Random(321),np.random.default_rng(321))
    assert metadata["multi_panel"]
    assert 2<=len(metadata["panels"])<=6
    assert len(masks)==len(metadata["curves"])==2*len(metadata["panels"])
    assert {curve["panel_id"] for curve in metadata["curves"]}=={
        panel["id"] for panel in metadata["panels"]
    }
    assert all(mask.getbbox() is not None for mask in masks)
