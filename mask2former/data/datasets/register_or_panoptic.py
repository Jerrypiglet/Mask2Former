'''
Adapted from mask2former/data/datasets/register_ade20k_panoptic.py

Removing the semseg labels (for semseg tasks)

Rui Zhu
'''

import json
import os
from pathlib import Path
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from datasets.prepare_or_pan_seg import OR_SEM_SEG_CATEGORIES, PALETTE
OR_46_CATEGORIES = []
for cat_id, cat_name in enumerate(OR_SEM_SEG_CATEGORIES):
    OR_46_CATEGORIES.append(
        {
            "name": cat_name,
            "id": cat_id,
            "isthing": cat_name not in ["floor_44", "ceiling_45"],
            "color": PALETTE[cat_id],
        }
    )



MetadataCatalog.get("or_panoptic_train").set(
    stuff_colors=PALETTE,
)

MetadataCatalog.get("or_panoptic_val").set(
    stuff_colors=PALETTE,
)


def load_or_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in tqdm(json_info["annotations"]):
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = str(Path(image_dir) / ann["image_file_name"])
        # label_file = os.path.join(gt_dir, ann["file_name"])
        pan_seg_file = str(Path(gt_dir) / ann["pan_seg_file_name"])
        assert Path(image_file).exists(), image_file
        assert Path(pan_seg_file).exists(), pan_seg_file
        
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": pan_seg_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    # assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    
    return ret


def register_or_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "or_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        # sem_seg_root (none): not used, to be consistent with
        #     `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_or_panoptic_json(
            panoptic_json, image_root, panoptic_root, metadata
        ),
    )
    
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        # json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_OR_PANOPTIC = {
    "or_panoptic_train": (
        "OpenRooms_public",
        "OpenRooms_public/im_pan_seg_rgb",
        "OpenRooms_public/im_pan_seg_rgb_train.json",
        # "OpenRooms_public/im_pan_seg_rgb_val.json",
    ), 
    "or_panoptic_val": (
        "OpenRooms_public",
        "OpenRooms_public/im_pan_seg_rgb",
        "OpenRooms_public/im_pan_seg_rgb_val.json",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in OR_46_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OR_46_CATEGORIES if k["isthing"] == 1]
    # stuff_classes = [k["name"] for k in OR_46_CATEGORIES if k["isthing"] == 0]
    # stuff_colors = [k["color"] for k in OR_46_CATEGORIES if k["isthing"] == 0]
    stuff_classes = [k["name"] for k in OR_46_CATEGORIES]
    stuff_colors = [k["color"] for k in OR_46_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors
    
    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(OR_46_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_or_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_OR_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_or_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_or_panoptic(_root)
