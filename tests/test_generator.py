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
    signatures={
        (tuple(curve["coefficients"]),json.dumps(curve["function_parameters"],sort_keys=True))
        for curve in related
    }
    assert len(signatures)==len(related)


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


def test_balanced_profile_has_readability_limits():
    profile = GeneratorConfig.from_json(
        Path(__file__).parents[1] / "configs" / "balanced_lineex_v5.json"
    )
    assert profile.max_curves <= 6
    assert profile.max_panels <= 3
    assert profile.max_curves_per_panel <= 3
    assert profile.curve_complexity <= 0.5
    assert profile.page_plot_min_fraction >= 0.55
    assert profile.page_layout_probability <= 0.05
    assert profile.multi_panel_probability <= 0.05
    assert profile.crop_probability <= 0.05


def test_dsc_mode_generates_composite_related_traces():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=320,
        height=240,
        supersample=1,
        min_curves=4,
        max_curves=4,
        max_points=240,
        legend_probability=0,
        annotations_probability=0,
        multi_panel_probability=0,
        dsc_page_layout_probability=0,
        seed=23,
    )
    generator=DatasetGenerator(cfg)
    image,masks,metadata=generator._render_base(random.Random(23),np.random.default_rng(23))
    assert image.size==(320,240)
    assert metadata["plot_domain"]=="dsc"
    assert metadata["dsc_layout"] in {"stacked","overlay"}
    assert len(masks)==len(metadata["curves"])==4
    assert all(curve["family"]=="dsc_trace" for curve in metadata["curves"])
    assert all(curve["function_parameters"]["events"] for curve in metadata["curves"])
    assert all(mask.getbbox() is not None for mask in masks)


def test_dsc_mode_can_render_one_pixel_curves():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=320,
        height=240,
        supersample=1,
        min_curves=6,
        max_curves=6,
        max_points=240,
        legend_probability=0,
        annotations_probability=0,
        multi_panel_probability=0,
        dsc_page_layout_probability=0,
    )
    generator=DatasetGenerator(cfg)
    found=False
    for seed in range(20):
        _,_,metadata=generator._render_base(random.Random(seed),np.random.default_rng(seed))
        if any(curve["line_width_px"]==1 for curve in metadata["curves"]):
            found=True
            break
    assert found


def test_dsc_dense_template_has_many_asymmetric_events():
    generator=DatasetGenerator(GeneratorConfig(plot_domain="dsc",dsc_max_events=18))
    template=generator._dsc_template(random.Random(91),dense=True)
    assert 7<=len(template["events"])<=18
    assert any(event["width_left"]!=event["width_right"] for event in template["events"])
    assert min(
        min(event["width_left"],event["width_right"])
        for event in template["events"]
    ) < 0.01


def test_dsc_mode_supports_multiple_panels():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=384,
        height=288,
        supersample=1,
        min_curves=1,
        max_curves=2,
        max_curves_per_panel=2,
        max_panels=3,
        max_points=180,
        multi_panel_probability=1,
        annotations_probability=0,
    )
    _,masks,metadata=DatasetGenerator(cfg)._render_base(
        random.Random(17),np.random.default_rng(17)
    )
    assert metadata["multi_panel"]
    assert metadata["plot_domain"]=="dsc"
    assert 2<=len(metadata["panels"])<=3
    assert masks
    assert metadata["dsc_panel_layout"] in {
        "top_bottom","left_right","staggered","top_focus","bottom_focus","three_rows"
    }
    assert all(panel["rendered_natively"] for panel in metadata["panels"])
    for panel in metadata["panels"]:
        x0,y0,x1,y1=panel["base_bbox"]
        assert panel["rendered_size_px"]==[x1-x0,y1-y0]
        assert 0<=x0<x1<=384 and 0<=y0<y1<=288


def test_dsc_config_profile_is_valid():
    profile=GeneratorConfig.from_json(Path(__file__).parents[1]/"configs"/"dsc_v1.json")
    assert profile.plot_domain=="dsc"
    assert profile.supersample>=2
    assert profile.markers_probability==0
    assert profile.dsc_dense_probability>0
    assert profile.dsc_max_events>=12
    assert profile.dsc_page_layout_probability>0
    assert profile.dsc_surrounding_text_probability>0
    assert profile.dsc_axis_font_max>=24
    assert profile.dsc_curve_label_font_max>=20
    assert profile.hard_negatives_probability>0


def test_dsc_direct_labels_cover_the_whole_curve_and_support_large_fonts():
    cfg=GeneratorConfig(
        plot_domain="dsc",width=640,height=480,supersample=1,
        min_curves=3,max_curves=3,max_points=320,multi_panel_probability=0,
        dsc_page_layout_probability=0,dsc_direct_labels_probability=1,
        dsc_curve_label_font_min=20,dsc_curve_label_font_max=20,
        dsc_annotation_font_min=20,dsc_annotation_font_max=20,
        annotations_probability=1,legend_probability=0,
        hard_negatives_probability=0,occlusion_probability=0,
        dsc_watermark_probability=0,
    )
    regions=set(); annotation_sizes=[]
    for seed in range(12):
        _,_,metadata=DatasetGenerator(cfg)._render_base(
            random.Random(seed),np.random.default_rng(seed)
        )
        for curve in metadata["curves"]:
            position=curve.get("label_position")
            assert position and position["font_size_px"]==20
            regions.add(position["region"])
        annotation_sizes.extend(item["font_size_px"] for item in metadata["annotations"])
    assert {"left","middle","right"}<=regions
    assert annotation_sizes and max(annotation_sizes)==20


def test_dsc_classic_obfuscations_are_recorded_and_removed_from_masks():
    cfg=GeneratorConfig(
        plot_domain="dsc",width=480,height=360,supersample=1,
        min_curves=2,max_curves=2,max_points=280,multi_panel_probability=0,
        dsc_page_layout_probability=0,dsc_direct_labels_probability=1,
        annotations_probability=0,legend_probability=0,
        hard_negatives_probability=1,occlusion_probability=1,
        dsc_watermark_probability=0,
    )
    _,masks,metadata=DatasetGenerator(cfg)._render_base(
        random.Random(117),np.random.default_rng(117)
    )
    types={item["type"] for item in metadata["occluders"]}
    assert "text_box_occlusion" in types
    assert types & {"reference_line","integration_baseline","onset_tangent"}
    assert metadata["hard_negative_count"]>=2
    assert all(mask.getbbox() is not None for mask in masks)


def test_dsc_page_layout_can_embed_a_small_plot_with_hard_negatives():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=640,
        height=480,
        supersample=1,
        min_curves=2,
        max_curves=2,
        max_points=300,
        multi_panel_probability=0,
        dsc_page_layout_probability=1,
        dsc_plot_min_fraction=.20,
        dsc_surrounding_text_probability=1,
        dsc_caption_probability=1,
        dsc_foreign_graphics_probability=1,
        dsc_watermark_probability=1,
        annotations_probability=0,
    )
    image,masks,metadata=DatasetGenerator(cfg)._render_base(
        random.Random(41),np.random.default_rng(41)
    )
    assert image.size==(640,480)
    assert metadata["page_layout"]
    assert metadata["plot_domain"]=="dsc"
    assert metadata["caption"]
    assert metadata["watermark"]
    assert metadata["hard_negative_count"]>=3
    assert any(item["type"]=="surrounding_text" for item in metadata["page_elements"])
    x0,y0,x1,y1=metadata["figure_bbox"]
    assert (x1-x0)*(y1-y0)<640*480
    assert len(masks)==2 and all(mask.getbbox() is not None for mask in masks)
    plot_x0,plot_y0,plot_x1,plot_y1=metadata["plot_bbox"]
    stroke_tolerance=5+max(curve["line_width_px"] for curve in metadata["curves"])
    for mask in masks:
        mask_x0,mask_y0,mask_x1,mask_y1=mask.getbbox()
        assert plot_x0-stroke_tolerance<=mask_x0<mask_x1<=plot_x1+stroke_tolerance
        assert plot_y0-stroke_tolerance<=mask_y0<mask_y1<=plot_y1+stroke_tolerance


def test_dsc_multipanel_can_be_combined_with_document_layers():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=640,
        height=480,
        supersample=1,
        min_curves=1,
        max_curves=2,
        max_curves_per_panel=2,
        max_panels=2,
        max_points=240,
        multi_panel_probability=1,
        dsc_multipanel_page_probability=1,
        dsc_surrounding_text_probability=1,
        dsc_caption_probability=1,
        dsc_foreign_graphics_probability=1,
        dsc_watermark_probability=1,
        annotations_probability=0,
    )
    image,masks,metadata=DatasetGenerator(cfg)._render_base(
        random.Random(73),np.random.default_rng(73)
    )
    assert image.size==(640,480)
    assert metadata["multi_panel"] and metadata["page_layout"]
    assert metadata["dsc_document_layout"]
    assert metadata["dsc_document_placement"] in {"top","bottom","left","right","center"}
    assert metadata["caption"] and metadata["watermark"]
    assert any(item["type"]=="surrounding_text" for item in metadata["page_elements"])
    figure_x0,figure_y0,figure_x1,figure_y1=metadata["figure_bbox"]
    assert (figure_x1-figure_x0)*(figure_y1-figure_y0)<640*480
    assert len(metadata["panels"])==2
    for panel in metadata["panels"]:
        x0,y0,x1,y1=panel["base_bbox"]
        assert figure_x0<=x0<x1<=figure_x1
        assert figure_y0<=y0<y1<=figure_y1
        assert panel["rendered_natively"]
    assert masks and all(mask.getbbox() is not None for mask in masks)


def test_three_panel_document_layout_keeps_panels_readable():
    cfg=GeneratorConfig(
        plot_domain="dsc",
        width=768,
        height=576,
        supersample=1,
        min_curves=1,
        max_curves=2,
        max_curves_per_panel=2,
        max_panels=3,
        max_points=240,
        multi_panel_probability=1,
        dsc_multipanel_page_probability=1,
        dsc_surrounding_text_probability=1,
        annotations_probability=0,
    )
    generator=DatasetGenerator(cfg)
    metadata=None
    for seed in range(30):
        _,masks,candidate=generator._render_base(random.Random(seed),np.random.default_rng(seed))
        if len(candidate["panels"])==3:
            metadata=candidate
            break
    assert metadata is not None
    assert masks
    for panel in metadata["panels"]:
        panel_width,panel_height=panel["rendered_size_px"]
        assert panel_width>=230
        assert panel_height>=115
