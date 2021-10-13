import os
import json

from .funcs import base_coco_format, filename_id_mapping, annotations_collections


class PartDpI:
    dp_I_to_part = {
        1: 'Torso',
        2: 'Torso',
        3: 'Right Hand',
        4: 'Left Hand',
        5: 'Left Foot',
        6: 'Right Foot',
        7: 'Upper Leg Right',
        8: 'Upper Leg Left',
        9: 'Upper Leg Right',
        10: 'Upper Leg Left',
        11: 'Lower Leg Right',
        12: 'Lower Leg Left',
        13: 'Lower Leg Right',
        14: 'Lower Leg Left',
        15: 'Upper Arm Left',
        16: 'Upper Arm Right',
        17: 'Upper Arm Left',
        18: 'Upper Arm Right',
        19: 'Lower Arm Left',
        20: 'Lower Arm Right',
        21: 'Lower Arm Left',
        22: 'Lower Arm Right',
        23: 'Head',
        24: 'Head',
    }
    part_to_dp_I = {}
    for key, value in dp_I_to_part.items():
        if value in part_to_dp_I:
            part_to_dp_I[value].append(key)
        else:
            part_to_dp_I[value] = [key]


def extract_box_from_coco_dp(
        json_path, out_json_path, need_dp_I=None, need_part=None,
):
    assert (need_dp_I is not None) ^ (need_part is not None)
    # 1. create need categories and index dp_I list
    categories = list()
    if need_dp_I is not None:
        for ndp_i in need_dp_I:
            assert ndp_i in range(1, 25), 'dp_I value out of range 1-24.'
            if PartDpI.dp_I_to_part[ndp_i] not in categories:
                categories.append(PartDpI.dp_I_to_part[ndp_i])
    else:
        need_dp_I = []
        for npart in need_part:
            if npart not in categories:
                categories.append(npart)
                assert npart in PartDpI.part_to_dp_I
                need_dp_I.append(PartDpI.part_to_dp_I[npart])

    # 2. load json files and extract each part infos
    jfiles = json.load(open(json_path))
    images = jfiles['images']
    # image id mapping
    filename_to_id, id_to_filename, id_lists = filename_id_mapping(images)

    annotations = jfiles['annotations']
    # collect annotations in same image
    collect_annos = annotations_collections(annotations)

    # 3. create base coco dict
    base_coco, label_to_categories = base_coco_format(
        need_categories=categories
    )

    # 4. extract infos
    img_id = 0
    anno_id = 0
    for idx, value in enumerate(images):
        id = value['id']
        img_info = value
        img_info['id'] = img_id
        img_id += 1
        base_coco['images'].append(
            img_info
        )

        for anno in collect_annos[id]:
            box = anno['bbox']
            x, y, w, h = box
            anno_dp_I = anno['dp_I']
            anno_dp_x = anno['dp_x']
            anno_dp_y = anno['dp_y']
            for ndi in need_dp_I:
                c_dp_x = [
                    x + anno_dp_x[int(di)] / 255 * x
                    for i, di in enumerate(anno_dp_I) if di == ndi
                ]
                c_dp_y = [
                    y + anno_dp_y[int(di)] / 255 * y
                    for i, di in enumerate(anno_dp_I) if di == ndi
                ]
                cx1, cy1 = min(c_dp_x), min(c_dp_y)
                cx2, cy2 = max(c_dp_x), max(c_dp_y)
                w, h = cx2 - cx1, cy2 - cy1
                c_annos = dict(
                    segmentation=[],
                    area=w * h,
                    iscrowd=0,
                    image_id=img_id - 1,
                    bbox=[cx1, cy1, w, h],
                    category_id=categories.index(PartDpI.dp_I_to_part[ndi]),
                    id=anno_id,
                )
                base_coco['annotations'].append(c_annos)
                anno_id += 1
    json.dump(
        base_coco,
        open(out_json_path, 'w'),
        indent=4,
    )
