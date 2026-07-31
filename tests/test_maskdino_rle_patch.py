from pathlib import Path

from training.patch_maskdino_rle import main


def test_maskdino_mapper_patch_is_idempotent(tmp_path: Path) -> None:
    mapper = (
        tmp_path
        / "maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py"
    )
    mapper.parent.mkdir(parents=True)
    mapper.write_text(
        """instances = utils.annotations_to_instances(annos, image_shape)
instances.gt_masks = PolygonMasks([])
gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
""",
        encoding="utf-8",
    )

    assert main(["--maskdino-root", str(tmp_path)]) == 0
    assert main(["--maskdino-root", str(tmp_path)]) == 0
    patched = mapper.read_text(encoding="utf-8")
    assert 'mask_format="bitmask"' in patched
    assert "BitMasks(torch.zeros" in patched
    assert "gt_masks = gt_masks.tensor" in patched
