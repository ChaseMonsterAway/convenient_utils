import os
import csv
import json
import random
import datetime
from abc import abstractmethod
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from lxml import etree


class BaseTransferToCoco:

    def __init__(
            self,
            outof_need_labels_ignore=False,
            outof_need_labels_background=False
    ) -> None:
        assert not outof_need_labels_background & outof_need_labels_ignore
        self.outlabel_ignore = outof_need_labels_ignore
        self.outlabel_background = outof_need_labels_background

    @abstractmethod
    def parse(self, file_path):
        """
        Return:
            extract_infos: list(
                [label, min_x, min_y, max_x, max_y]
            )
            image_height: int
            image_width: int
            image_path: str
        """
        pass

    def update_coco_dict(
            self,
            coco_dict,
            file_path,
            image_id,
            annotation_id,
            label_to_categories,
            min_area=None,
    ):
        parse_res = self.parse(
            file_path=file_path,
        )
        if parse_res is None:
            return None
        extract_infos, image_height, image_width, image_path = parse_res
        coco_dict['images'].append(
            dict(
                file_name=image_path,
                height=image_height,
                width=image_width,
                id=image_id,
            )
        )
        for info in extract_infos:
            x, y, w, h = info[1], info[2], info[3] - info[1], info[4] - info[2]
            area = w * h
            iscrowd = 0
            label = info[0]
            if label not in label_to_categories:
                if self.outlabel_background:
                    continue
                if self.outlabel_ignore:
                    iscrowd = 1
                    label_to_categories[label_to_categories] = max(
                        list(label_to_categories.values())
                    ) + 1
            if min_area is not None:
                if area < min_area:
                    iscrowd = 1

            coco_dict['annotations'].append(
                dict(
                    segmentation=list(),
                    area=area,
                    iscrowd=iscrowd,
                    image_id=image_id,
                    bbox=[
                        x, y, w, h
                    ],
                    category_id=label_to_categories[info[0]],
                    id=annotation_id,
                ),
            )
            annotation_id += 1
        image_id += 1

        return coco_dict, image_id, annotation_id

    def base_coco_format(self, need_categories, super_category='coco'):
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

    def transfer(
            self,
            file_dir,
            out_coco_json_path,
            need_categories=None,
            min_area=None,
    ):
        image_id = 0
        annotation_id = 0
        assert need_categories is not None, 'You should specify the\
            needed categories.'
        coco_dict, label_to_categories = self.base_coco_format(need_categories)
        if not os.path.exists(os.path.dirname(out_coco_json_path)):
            os.makedirs(os.path.dirname(out_coco_json_path))
        for file_name in tqdm(os.listdir(file_dir)):
            res = self.update_coco_dict(
                coco_dict=coco_dict,
                file_path=os.path.join(
                    file_dir, file_name
                ),
                image_id=image_id,
                annotation_id=annotation_id,
                label_to_categories=label_to_categories,
                min_area=min_area,
            )
            if res is not None:
                coco_dict, image_id, annotation_id = res

        json.dump(
            coco_dict,
            open(out_coco_json_path, 'w'),
            indent=4,
        )


class GuangdongDianWangTransferToCoco(BaseTransferToCoco):
    def __init__(
            self,
            img_root=str,
            split_ratio=0.2,
            remain=True,
            seed=0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.img_root = img_root
        self.split_ratio = split_ratio
        self.seed = seed
        self.remain = remain

    def set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def select(self, candidate: dict, ratio):
        keys = list(candidate.keys())
        select_keys = random.sample(keys, int(len(keys) * ratio))
        if self.remain:
            return {
                key: value for key, value in candidate.items() if
                key not in select_keys
            }

        return {
            key: value for key, value in candidate.items() if
            key in select_keys
        }

    def parse(self, file_path):
        infos = file_path
        assert isinstance(infos, tuple)
        filename = infos[0]
        img = cv2.imread(
            os.path.join(
                self.img_root, filename
            )
        )
        if img is None:
            print('Empty file {}'.format(file_path))
            return None
        height, width = img.shape[:2]
        annos = []
        for key, value in infos[1].items():
            for v in value:
                annos.append(
                    [key, v[0], v[1], v[2], v[3]]
                )
        return annos, height, width, filename

    def extract_annotations(self, annotation: list):
        img_name = os.path.basename(annotation[4])
        metas = eval(
            annotation[5].replace('false', 'False').replace('true', 'True')
        )
        items = metas['items']
        annos = dict()
        for item in items:
            label = item['labels']['标签']
            if label not in annos:
                annos[label] = []
            meta = item['meta']
            bbox = meta['geometry']
            start_x, start_y = min(bbox[0::2]), min(bbox[1::2])
            end_x, end_y = max(bbox[0::2]), max(bbox[1::2])
            annos[label].append(
                [start_x, start_y, end_x, end_y]
            )

        return img_name, annos

    def parse_csv(self, csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return list(reader)

    def imgname_anno_dict(self, csv_file_path):
        self.set_seed()
        infos = self.parse_csv(csv_file_path)
        res = dict()
        for info in infos:
            imgname, annos = self.extract_annotations(info)
            res[imgname] = annos
        res = self.select(res, self.split_ratio)

        return res

    def transfer(
            self,
            file_dir,
            out_coco_json_path,
            need_categories,
            min_area=None,
    ):
        image_id = 0
        annotation_id = 0
        assert need_categories is not None, 'You should specify the\
            needed categories.'
        coco_dict, label_to_categories = self.base_coco_format(need_categories)
        if not os.path.exists(os.path.dirname(out_coco_json_path)):
            os.makedirs(os.path.dirname(out_coco_json_path))
        for key, value in self.imgname_anno_dict(csv_file_path=file_dir).items():
            res = self.update_coco_dict(
                coco_dict=coco_dict,
                file_path=(key, value),
                image_id=image_id,
                annotation_id=annotation_id,
                label_to_categories=label_to_categories,
                min_area=min_area,
            )
            if res is not None:
                coco_dict, image_id, annotation_id = res
        json.dump(
            coco_dict,
            open(out_coco_json_path, 'w'),
            indent=4,
        )


class XmlTransferToCoco(BaseTransferToCoco):
    def __init__(
            self,
            min_size=None,
            img_root=None,
            filter_gray=False,
            outof_need_labels_ignore=False,
            outof_need_labels_background=False,
    ) -> None:
        super().__init__(
            outof_need_labels_ignore=outof_need_labels_ignore,
            outof_need_labels_background=outof_need_labels_background
        )
        self.min_size = min_size
        self.img_root = img_root
        self.filter_gray = filter_gray

    def update_img_root(self, root):
        assert isinstance(root, str)
        assert os.path.exists(root)
        self.img_root = root

    def get_img_posix(self, dirs, filename):
        possible_posix = [
            '.jpg', '.png', '.bmp', '.jpeg'
        ]
        matched = []
        for posix in possible_posix:
            if os.path.exists(
                    os.path.join(
                        dirs, filename + posix
                    )
            ):
                matched.append(
                    os.path.join(
                        dirs, filename + posix
                    )
                )
        if len(matched) == 1:
            return True, matched[0]

        return False, matched

    def parse(self, file_path):
        tree_root = etree.parse(file_path)
        filename = tree_root.find("filename").text
        if '.'.join(filename.split('.')[:-1]) != os.path.basename(
                file_path.replace('\\', '/')
        ):
            filename = '.'.join(
                os.path.basename(file_path.replace('\\', '/')).split('.')[:-1]
            )
            if self.img_root is not None:
                flag, path = self.get_img_posix(self.img_root, filename)
                if not flag:
                    print(path)
                    return None
                filename = os.path.basename(path.replace('\\', '/'))

        if self.img_root is not None:
            if not os.path.exists(
                    os.path.join(
                        self.img_root, os.path.basename(filename)
                    )
            ):
                print('Not exist {}'.format(
                    os.path.join(
                        self.img_root, os.path.basename(filename)
                    )
                ))
                return None
        sizes = tree_root.find('size')
        height = int(sizes.find('width').text)
        width = int(sizes.find('height').text)
        depth = int(sizes.find('depth').text)
        if self.filter_gray and depth == 1:
            return None

        if self.min_size is not None:
            if isinstance(self.min_size, tuple):
                if height < self.min_size[0] or width < self.min_size[1]:
                    return None
            elif isinstance(self.min_size, (int, float)):
                if max(height, width) < self.min_size:
                    return None
        annos = []
        for object in tree_root.findall("object"):
            label = object.find("name").text
            bbox = object.find("bndbox")
            bbox_list = []
            bbox_list.append(int(round(float(bbox.find("xmin").text), 2)))
            bbox_list.append(int(round(float(bbox.find("ymin").text), 2)))
            bbox_list.append(int(round(float(bbox.find("xmax").text), 2)))
            bbox_list.append(int(round(float(bbox.find("ymax").text), 2)))
            annos.append(
                [label, *bbox_list]
            )
        return annos, height, width, filename


class CombineCoCo:
    def __init__(
            self,
            label_mapping=[
                dict(),
            ],
            cocofiles=[],
    ) -> None:
        assert len(label_mapping) == len(cocofiles)
        self.label_mapping = label_mapping
        self.cocofiles = cocofiles
        self.last_img_id = -1
        self.last_anno_id = -1

    def export(self, out_json):
        assert out_json.endswith('.json')
        container = self.combine()
        json.dump(
            container,
            open(out_json, 'w'),
            indent=4,
        )

    def create_container(self):
        return dict(
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
            categories=list(),
        )

    def combine(self):
        container = self.create_container()
        for idx, cocofile in enumerate(self.cocofiles):
            json_file = json.load(
                open(cocofile)
            )
            label_mapping = self.label_mapping[idx]
            images = json_file['images']
            annotations = json_file['annotations']
            categories = json_file['categories']
            self.update(
                container,
                images,
                annotations,
                categories,
                label_mapping
            )
        return container

    def update(
            self,
            container,
            images,
            annotations,
            categories,
            label_mapping
    ):
        if len(container['images']) == 0:
            # append images
            container['images'] = images
            # update categories, base only change name
            new_categories = []
            for idx, cate in enumerate(categories):
                new_category = cate.copy()
                if cate['name'] in label_mapping:
                    map_name = label_mapping[cate['name']]
                    new_category['name'] = label_mapping[cate['name']]
                new_categories.append(new_category)
            container['annotations'] = annotations
            container['categories'] = new_categories
        else:
            category_id_mapping = self.gerate_category_id_mapping(
                container['categories'],
                categories,
                label_mapping=label_mapping,
            )
            for image in images:
                image['id'] = image['id'] + 1 + self.last_img_id
            for annotation in annotations:
                annotation['image_id'] = annotation['image_id'] + 1 + \
                                         self.last_img_id
                annotation['id'] = annotation['id'] + 1 + self.last_anno_id
                annotation['category_id'] = category_id_mapping[
                    annotation['category_id']
                ]
            container['images'] += images
            container['annotations'] += annotations
        #  record the last index of images and annotations
        self.last_img_id = container['images'][-1]['id']
        if container['annotations']:
            self.last_anno_id = container['annotations'][-1]['id']
        else:
            self.last_anno_id = -1
        return container

    def gerate_category_id_mapping(
            self,
            existing_categories: list,
            current_categories: list,
            label_mapping: dict,
    ):
        category_id_mapping = dict()
        new_current_categories = self.fix_category_name(
            current_categories, label_mapping
        )
        existing_name, existing_name_to_id, m_id = self.collect_category_name(
            existing_categories
        )
        new_c_name, new_c_name_to_id, _ = self.collect_category_name(
            new_current_categories
        )
        for idx, c_name in enumerate(new_c_name):
            if c_name in existing_name:
                category_id_mapping[new_c_name_to_id[c_name]] = \
                    existing_name_to_id[c_name]
            else:
                # update existing category
                existing_name.append(c_name)
                existing_name_to_id[c_name] = m_id + 1
                m_id += 1
                ncate = new_current_categories[idx].copy()
                ncate['id'] = m_id
                existing_categories.append(ncate)

                category_id_mapping[
                    new_c_name_to_id[c_name]
                ] = m_id

        return category_id_mapping

    def collect_category_name(self, categories):
        names, name_to_id = list(), dict()
        max_id = -1
        for cate in categories:
            names.append(cate['name'])
            c_id = cate['id']
            max_id = max(max_id, c_id)
            name_to_id[cate['name']] = c_id

        return names, name_to_id, max_id

    def fix_category_name(
            self,
            categories,
            label_mapping,
    ):
        new_categories = []
        for idx, cate in enumerate(categories):
            new_category = cate.copy()
            if cate['name'] in label_mapping:
                new_category['name'] = label_mapping[cate['name']]
            new_categories.append(new_category)

        return new_categories
