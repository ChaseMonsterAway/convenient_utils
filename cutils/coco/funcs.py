import csv
import datetime
import json
import os
import random
from abc import abstractmethod
from collections import defaultdict

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm


def base_coco_format(need_categories, super_category='coco'):
    categories = [
        dict(
            supercategory=super_category,
            id=idx + 1,
            name=category
        )
        for idx, category in enumerate(need_categories)
    ]
    label_to_categories = {
        category['name']: category['id'] for category in categories
    }
    coco_dict = dict(
        info=dict(
            description='',
            url='',
            version='',
            year=2021,
            contributor='',
            date_created=datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        license=list(),
        images=list(),
        annotations=list(),
        categories=categories,
    )

    return coco_dict, label_to_categories


def remove_duplicate_label(json_file):
    jfiles = json.load(open(json_file))
    annotations = jfiles['annotations']
    categories = jfiles['categories']
    new_categories = []
    existing_label_name = []
    cate_id = 1
    old_id_to_new_id = dict()
    for idx, cate in enumerate(categories):
        base_cate = {
            "supercategory": "coco",
            "id": 1,
            "name": None
        }
        name = cate['name']
        if name in existing_label_name:
            old_id_to_new_id[idx + 1] = existing_label_name.index(name) + 1
        else:
            existing_label_name.append(name)
            base_cate['name'] = name
            base_cate['id'] = cate_id
            old_id_to_new_id[idx + 1] = cate_id
            cate_id += 1
            new_categories.append(base_cate)
    jfiles['categories'] = new_categories
    for anno in tqdm(annotations):
        old__id = anno['category_id']
        new_id = old_id_to_new_id[old__id]
        anno['category_id'] = new_id
    newfilename = 'new_' + os.path.basename(json_file)
    json.dump(
        jfiles,
        open(
            os.path.join(os.path.dirname(json_file), newfilename), 'w'
        ),
        indent=4,
    )


def filename_id_mapping(imgs_info: list):
    """
    create id mapping among filename and img id.
    """
    filename_to_id = dict()
    id_to_filename = dict()
    id_lists = list()
    for img in imgs_info:
        filename = img['file_name']
        img_id = img['id']
        id_lists.append(img_id)
        filename_to_id[filename] = img_id
        id_to_filename[img_id] = filename
    return filename_to_id, id_to_filename, id_lists


def annotations_collections(annotations: list):
    """
    collect annotations together if the annotations appear in same image.
    """
    anno_collections = defaultdict(list)
    for anno in annotations:
        image_id = anno['image_id']
        anno_collections[image_id].append(anno)
    return anno_collections


def split_coco_trainval(json_path, train_ratio, image_based=True):
    jfiles = json.load(open(json_path))
    images = jfiles['images']
    annotations = jfiles['annotations']
    collect_annos = annotations_collections(annotations)

    if image_based:
        new_images = dict(
            train=[],
            val=[],
        )
        new_annotations = dict(
            train=[],
            val=[],
        )
        train_nums = int(len(images) * train_ratio)
        ids = [i for i in range(len(images))]
        random.shuffle(ids)
        new_train_img_id = 0
        new_train_anno_id = 0
        new_val_img_id = 0
        new_val_anno_id = 0
        for idx, img_id in enumerate(ids):
            if idx < train_nums:
                c_image = images[img_id]
                c_annotations = collect_annos[img_id]
                c_image['id'] = new_train_img_id
                new_images['train'].append(c_image)
                for anno in c_annotations:
                    anno['image_id'] = new_train_img_id
                    anno['id'] = new_train_anno_id
                    new_train_anno_id += 1
                new_train_img_id += 1
            else:
                c_image = images[img_id]
                c_annotations = collect_annos[img_id]
                c_image['id'] = new_val_img_id
                new_images['val'].append(c_image)
                for anno in c_annotations:
                    anno['image_id'] = new_val_img_id
                    anno['id'] = new_val_anno_id
                    new_val_anno_id += 1
                new_val_img_id += 1
        train_coco = base_coco_format(
            [cate['name'] for cate in jfiles['categories']],
        )
        train_coco['images'] = new_images['train']
        train_coco['annotations'] = new_annotations['train']
        val_coco = base_coco_format(
            [cate['name'] for cate in jfiles['categories']]
        )
        val_coco['images'] = new_images['val']
        val_coco['annotations'] = new_annotations['val']
    else:
        # annotations based
        raise ValueError('Currently, only support image based split.')

    return train_coco, val_coco


def split_coco_trainval_save(
        json_path, train_ratio, save_dir, image_based=True
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_coco, val_coco = split_coco_trainval(json_path, train_ratio, image_based)
    json.dump(
        train_coco,
        open(os.path.join(save_dir, 'train.json'), 'w'),
        indent=4,
    )
    json.dump(
        val_coco,
        open(os.path.join(save_dir, 'val.json', 'w')),
        indent=4,
    )


def expand(json_path, out_json, expand_ratio=0.1):
    jfiles = json.load(open(json_path))
    images = jfiles['images']
    annos = jfiles['annotations']
    _, _, id_lists = filename_id_mapping(images)
    for anno in annos:
        img_id = anno['image_id']
        height, width = images[id_lists.index(img_id)]
        box = anno['bbox']
        area = anno['area']
        x, y, w, h = box
        x = max(0, x - expand_ratio * w)
        y = max(0, y - expand_ratio * h)
        w = min(width - 1, x + w * (1 + 2 * expand_ratio)) - x
        h = min(height - 1, y + h * (1 + 2 * expand_ratio)) - y
        anno['bbox'] = [x, y, w, h]
        anno['area'] = w * h
    json.dump(
        jfiles,
        open(out_json, 'w'),
        indent=4,
    )
