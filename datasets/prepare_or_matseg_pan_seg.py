#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
from collections import Counter

import numpy as np
from tqdm import tqdm
from panopticapi.utils import IdGenerator, save_json
from PIL import Image
import cv2
from pathlib import Path
import sys, os
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.utils_io import load_img
from utils.utils_misc import red, yellow
from utils.utils_or.utils_or_io import load_matseg
from panopticapi import utils as panopticapi_utils

project_root_path = Path('/home/ruizhu/Documents/Projects/matsegnerf')
or_root = Path('/data/OpenRooms_public/')
im_matseg_pan_seg_rgb_root = Path('/data/OpenRooms_public/im_matseg_pan_seg_rgb')
frame_list_root = project_root_path / 'data/openrooms/public'

OR_MATSEG_1_CATEGORIES = [
    {
        "name": 'or_matseg_instance',
        "id": 0,
        # "isthing": False,
        # "isstuff": True, 
        "isthing": True,
        "isstuff": False, 
        "color": [120, 120, 120],
    }
]

def get_im_info_list(split):
    frame_list_path = frame_list_root / ('%s.txt'%split)
    assert frame_list_path.exists(), frame_list_path
    im_info_list = []
    with open(frame_list_path, 'r') as f:
        frame_list = f.read().splitlines()
        # print(len(frame_list), frame_list[0])
        for frame_info in tqdm(frame_list):
            scene_name, frame_id, im_sdr_file, imsemLabel_path = frame_info.split(' ')
            meta_split, scene_name_, im_sdr_name = im_sdr_file.split('/')
            assert scene_name == scene_name_
            assert im_sdr_name.split('.')[0].split('_')[1] == str(frame_id)
                    
            scene_path = or_root / meta_split / scene_name
            im_sdr_path = scene_path / 'im_sdr' / f'im_{frame_id}.png'
            assert im_sdr_path.exists(), im_sdr_path
            
            scene_path_semantics = or_root / meta_split.replace('DiffMat', '').replace('DiffLight', '') / scene_name
            matseg_path = scene_path_semantics / 'matseg' / f'imcadmatobj_{frame_id}.dat'
            # assert matseg_path.exists(), str(matseg_path) + '+' + str(im_sdr_path)
            if not matseg_path.exists():
                print(yellow('matseg_path does not exist:'), matseg_path)
                continue

            im_info_dict = {
                'im_sdr_path': im_sdr_path,
                'matseg_path': matseg_path,
                'meta_split': meta_split,
                'scene_name': scene_name,
                'frame_id': frame_id,
            }
            
            im_info_list.append(im_info_dict)
            
    print('[%s-%s] im_info_list: len %d'%(split, str(frame_list_path), len(im_info_list)))
    
    return im_info_list


if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

    # for split in ['train', 'val']:
    # for split in ['val']:
    for split in ['train']:
        
        image_dir = os.path.join(dataset_dir, f"OpenRooms_public")

        # folder to store panoptic PNGs
        out_folder = os.path.join(dataset_dir, f"OpenRooms_public/im_matseg_pan_seg_rgb/")
        # json with segmentations information
        out_file = os.path.join(dataset_dir, f"OpenRooms_public/im_matseg_pan_seg_rgb_{split}.json")

        if not os.path.isdir(out_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
            os.mkdir(out_folder)

        # # json config
        # config_file = "datasets/ade20k_instance_imgCatIds.json"
        # with open(config_file) as f:
        #     config = json.load(f)

        categories_dict = {cat["id"]: cat for cat in OR_MATSEG_1_CATEGORIES}

        panoptic_json_categories = OR_MATSEG_1_CATEGORIES[:]
        panoptic_json_images = []
        panoptic_json_annotations = []

        # get all images 
        im_info_list = get_im_info_list(split)
        
        # process matseg
        for image_id, im_info_dict in tqdm(enumerate(im_info_list)):
            panoptic_json_annotation = {}
            
            filename = str(im_info_dict['im_sdr_path'])
            height, width = load_img(im_info_dict['im_sdr_path']).shape[:2]
            # print(height, width)

            im_matseg_pan_seg_rgb_folder = im_matseg_pan_seg_rgb_root / (im_info_dict['meta_split'].replace('DiffMat', '').replace('DiffLight', '')) / im_info_dict['scene_name']
            im_matseg_pan_seg_rgb_folder.mkdir(parents=True, exist_ok=True)
            im_matseg_pan_seg_rgb_path = im_matseg_pan_seg_rgb_folder / ('im_matseg_pan_seg_rgb_%s.png'%im_info_dict['frame_id'])
            
            # if im_matseg_pan_seg_rgb_path.exists():
            #     continue
            
            print(im_matseg_pan_seg_rgb_path)
            im_matseg_dict = load_matseg(im_info_dict['matseg_path'])
            if im_matseg_dict is None:
                print(red('im_matseg_dict is None (something wrong with the matseg label: %s)'%str(im_info_dict['matseg_path'])))
                continue
            
            mat_aggre_map = im_matseg_dict['mat_aggre_map']
            num_mat_masks = im_matseg_dict['num_mat_masks']
            
            im_matseg_pan_seg_ids = np.zeros((height, width), dtype=np.int16)
            segments_info = []
            
            for mask_id in np.arange(1, num_mat_masks+1):
                # for num_mat_mask in np.arange(1, 2):
                mask_single = mat_aggre_map == mask_id
                # print(np.sum(mask_single))
                # if np.sum(mask_single) < 500:
                #     continue
                
                # assert np.amax(mask_id) < 256
                im_matseg_pan_seg_ids[mask_single] = mask_id
                area = np.sum(mask_single)  # segment area computation
                segments_info.append({
                    'id': int(mask_id),
                    'category_id': 0,
                    'iscrowd': 0,
                    # 'isthing': 0,
                    # 'isstuff': 1,
                    'isthing': 1,
                    'isstuff': 0,
                    'area': int(area),
                })
                
            panoptic_json_annotation = {
                "image_id": image_id,
                "file_name": str(im_matseg_pan_seg_rgb_path.relative_to(or_root/'im_matseg_pan_seg_rgb')), 
                "image_file_name": str(im_info_dict['im_sdr_path'].relative_to(or_root)), 
                # "file_name": str(im_matseg_pan_seg_rgb_path.relative_to(or_root/'im_matseg_pan_seg_rgb')), 
                "pan_seg_file_name": str(im_matseg_pan_seg_rgb_path.relative_to(or_root/'im_matseg_pan_seg_rgb')), 
                "segments_info": segments_info,
            }
            panoptic_json_annotations.append(panoptic_json_annotation)

            panoptic_json_image = {}
            panoptic_json_image["id"] = image_id
            panoptic_json_image["file_name"] = str(im_info_dict['im_sdr_path'].relative_to(or_root))
            panoptic_json_image["width"] = width
            panoptic_json_image["height"] = height
            panoptic_json_images.append(panoptic_json_image)

            # if not im_matseg_pan_seg_rgb_path.exists():
            # print('WRITTT ->', str(im_matseg_pan_seg_rgb_path))
            im_matseg_pan_seg_rgb = panopticapi_utils.id2rgb(im_matseg_pan_seg_ids)
            cv2.imwrite(str(im_matseg_pan_seg_rgb_path), im_matseg_pan_seg_rgb[:, :, ::-1])
            # else:
            #     print('READDDD: ', str(im_matseg_pan_seg_rgb_path))
            #     im_matseg_pan_seg_rgb = cv2.imread(str(im_matseg_pan_seg_rgb_path))[::-1]
            #     pass
            
        # save json
        d = {
            "images": panoptic_json_images,
            "annotations": panoptic_json_annotations,
            "categories": panoptic_json_categories,
        }

        save_json(d, out_file)
        print('JSON faved to', out_file)
