from .create import (
    XmlTransferToCoco, GuangdongDianWangTransferToCoco, CombineCoCo
)
from .funcs import (
    base_coco_format, split_coco_trainval,
    split_coco_trainval_save, annotations_collections, filename_id_mapping,
    remove_duplicate_label,
)
from .transfer import extract_box_from_coco_dp
