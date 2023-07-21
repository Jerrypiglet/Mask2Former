#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
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
from utils.utils_or.utils_or_vis import vis_index_map
from panopticapi import utils as panopticapi_utils

project_root_path = Path('/home/ruizhu/Documents/Projects/matsegnerf')
or_root = Path('/data/OpenRooms_public/')
im_pan_seg_rgb_root = Path('/data/OpenRooms_public/im_pan_seg_rgb')
frame_list_root = project_root_path / 'data/openrooms/public'

IF_DEBUG = False
# excludes_scene_list_log_path = frame_list_root / 'exclude_semseg_log.txt'
excludes_scene_list = []

OR_SEM_SEG_CATEGORIES = [
    # "unlabeled", 
    "curtain_1", 
    "bike_2", 
    "washing_machine_3", 
    "table_4", 
    "desk_5", 
    "pool_table_6", 
    "counter_7", 
    "dishwasher_8", 
    "bowl_9", 
    "bookshelf_10", 
    "sofa_11", 
    "speaker_12", 
    "trash_bin_13", 
    "piano_14", 
    "guitar_15", 
    "pillow_16", 
    "jar_17", 
    "bed_18", 
    "bottle_19", 
    "clock_20", 
    "chair_21", 
    "computer_keyboard_22", 
    "monitor_23", 
    "whiteboard_24", 
    "bathtub_25", 
    "stove_26", 
    "microwave_27", 
    "file_cabinet_28", 
    "flowerpot_29", 
    "cap_30", 
    "window_31", 
    "lamp_32", 
    "telephone_33", 
    "printer_34", 
    "basket_35", 
    "faucet_36", 
    "bag_37", 
    "laptop_38", 
    "can_39", 
    "bench_40", 
    "door_41", 
    "cabinet_42", 
    
    "wall_43", 
    "floor_44", 
    "ceiling_45", 
]

PALETTE = [
    # [0, 0, 0], 
    [174, 199, 232], 
    [152, 223, 138], 
    [31, 119, 180], 
    [255, 187, 120], 
    [188, 189, 34], 
    [140, 86, 75], 
    [255, 152, 150], 
    [214, 39, 40], 
    [197, 176, 213], 
    [148, 103, 189], 
    [196, 156, 148], 
    [23, 190, 207], 
    [178, 76, 76], 
    [247, 182, 210], 
    [66, 188, 102], 
    [219, 219, 141], 
    [140, 57, 197], 
    [202, 185, 52], 
    [51, 176, 203], 
    [200, 54, 131], 
    [92, 193, 61], 
    [78, 71, 183], 
    [172, 114, 82], 
    [255, 127, 14], 
    [91, 163, 138], 
    [153, 98, 156], 
    [140, 153, 101], 
    [158, 218, 229], 
    [100, 125, 154], 
    [178, 127, 135], 
    [120, 185, 128], 
    [146, 111, 194], 
    [44, 160, 44], 
    [112, 128, 144], 
    [96, 207, 209], 
    [227, 119, 194], 
    [213, 92, 176], 
    [94, 106, 211], 
    [82, 84, 163], 
    [100, 85, 144], 
    [0, 0, 230], 
    [119, 11, 32], 
    [0, 80, 100], 
    [102, 102, 156], 
    [190, 153, 153], 
    ]

def get_exclude_scenes(scene_list_path):
    assert scene_list_path.exists(), scene_list_path
    with open(scene_list_path, 'r') as f:
        scene_list = f.read().splitlines()
    return [(_.split('/')[0], _.split('/')[1]) for _ in scene_list]
    
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

            semseg_path = scene_path_semantics / 'semseg' / f'imsemLabel_{frame_id}.npy'
            if not semseg_path.exists():
                print(yellow('semseg_path does not exist:'), semseg_path)
                continue
            
            immask_path = scene_path_semantics / 'mask' / f'immask_{frame_id}.png'
            if not immask_path.exists():
                print(yellow('immask_path does not exist:'), immask_path)
                continue
            
            im_info_dict = {
                'im_sdr_path': im_sdr_path,
                'matseg_path': matseg_path,
                'semseg_path': semseg_path,
                'immask_path': immask_path,
                'meta_split': meta_split,
                'meta_split_semantics': meta_split.replace('DiffMat', '').replace('DiffLight', ''),
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
        out_folder = os.path.join(dataset_dir, f"OpenRooms_public/im_pan_seg_rgb/")
        # json with segmentations information
        out_file = os.path.join(dataset_dir, f"OpenRooms_public/im_pan_seg_rgb_{split}.json")

        if not os.path.isdir(out_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
            os.mkdir(out_folder)

        OR_46_CATEGORIES = []
        for cat_id, cat_name in enumerate(OR_SEM_SEG_CATEGORIES):
            OR_46_CATEGORIES.append(
                {
                    "name": cat_name,
                    "id": cat_id,
                    "isthing": cat_name not in ["wall_43", "floor_44", "ceiling_45"],
                    "color": PALETTE[cat_id],
                }
            )
        categories_dict = {cat["id"]: cat for cat in OR_46_CATEGORIES}

        panoptic_json_categories = OR_SEM_SEG_CATEGORIES[:]
        panoptic_json_images = []
        panoptic_json_annotations = []

        # get all images 
        im_info_list = get_im_info_list(split)
        # excludes_scene_list = get_exclude_scenes(frame_list_root / 'exclude_semseg.txt')
        # im_info_list = [_ for _ in im_info_list if (_['meta_split_semantics'], _['scene_name']) not in excludes_scene_list]
        if IF_DEBUG:
            im_info_list = im_info_list[1:2]
        
        # process matseg
        for image_id, im_info_dict in tqdm(enumerate(im_info_list)):
            panoptic_json_annotation = {}
            
            filename = str(im_info_dict['im_sdr_path'])
            height, width = load_img(im_info_dict['im_sdr_path']).shape[:2]

            im_pan_seg_rgb_folder = im_pan_seg_rgb_root / (im_info_dict['meta_split'].replace('DiffMat', '').replace('DiffLight', '')) / im_info_dict['scene_name']
            im_pan_seg_rgb_folder.mkdir(parents=True, exist_ok=True)
            im_pan_seg_rgb_path = im_pan_seg_rgb_folder / ('im_pan_seg_rgb_%s.png'%im_info_dict['frame_id'])
            
            # if im_pan_seg_rgb_path.exists(): continue
            
            # load objs from matseg labels
            print(im_pan_seg_rgb_path)
            im_matseg_dict = load_matseg(im_info_dict['matseg_path'])
            if im_matseg_dict is None:
                print(red('im_matseg_dict is None (something wrong with the matseg label: %s)'%str(im_info_dict['matseg_path'])))
                continue
            raw_masks = im_matseg_dict['raw_masks']
            obj_idx_map = raw_masks[:, :, 2] # 3rd channel: object INDEX map
            
            # load semseg labels
            im_semseg = load_img(im_info_dict['semseg_path'], ext='npy').astype('uint8')
            
            # load object masks
            im_mask = load_img(im_info_dict['immask_path'], ext='png')[:, :, 0] / 255. # [0., 1.]
            seg_obj = im_mask > 0.9

            im_pan_seg_ids = np.zeros((height, width), dtype=np.int16) + 255
            segments_info = []
            
            # iterate over all instances, figure out thing/stuff classes and assign panoptic ids
            mask_id = 0
            IF_SKIP = False
            
            for obj_idx in np.unique(obj_idx_map):
                obj_mask = obj_idx_map == obj_idx
                obj_semseg = im_semseg[obj_mask]
                im_semseg_masked = im_semseg.copy()
                im_semseg_masked[~obj_mask] = 0
                im_semseg_masked[~seg_obj] = 0
                
                # visualize         
                # if IF_DEBUG:
                #     plt.figure()
                #     plt.subplot(131)
                #     # plt.imshow(im_sdr)
                #     ax = plt.subplot(132)
                #     ax.set_title('obj_idx: %d' % obj_idx)
                #     plt.imshow(obj_idx_map == obj_idx)
                #     ax = plt.subplot(133)
                #     plt.imshow(vis_index_map(im_semseg_masked))
                #     plt.show()
                    
                valid_obj_semseg_single_list = []
                for obj_semseg_single in np.unique(obj_semseg):
                    mask_single = im_semseg_masked==obj_semseg_single
                    
                    ratio_pixels_obj = float(np.sum(mask_single)) / float(np.sum(obj_mask)) * 100.
                    if ratio_pixels_obj < 5: continue
                    if np.sum(mask_single) < 1000: continue # [!!!] ignore extremely small objects, or fantom objects due to aliasing (e.g. https://i.imgur.com/OfAAIft.png)
                    valid_obj_semseg_single_list.append(obj_semseg_single)
                    
                    assert mask_id < 255
                    im_pan_seg_ids[mask_single] = mask_id
                    area = np.sum(mask_single)  # segment area computation
                    
                    segments_info.append({
                        'id': int(mask_id),
                        'category_id': int(obj_semseg_single - 1), # [!!!] now 0-44 is valid, 255 is ignored
                        'iscrowd': 0,
                        'isthing': obj_semseg_single not in [43, 44, 45],
                        'isstuff': obj_semseg_single in [43, 44, 45],
                        'area': int(area),
                    })
                    
                    # if IF_DEBUG:
                    #     plt.figure(figsize=(15, 8))
                    #     plt.title('obj_idx %d - %s, ratio of entire object: %.2f%%' % (obj_idx, OR_SEM_SEG_CATEGORIES[obj_semseg_single], ratio_pixels_obj))
                    #     plt.imshow(mask_single)
                    #     plt.show()
                    #     for _ in np.unique(im_pan_seg_ids):
                    #         plt.figure()
                    #         plt.imshow(im_pan_seg_ids == _)
                    #         plt.title('id: {}'.format(_))
                    #         plt.show()

                        
                    mask_id += 1
                
                # if len(valid_obj_semseg_single_list) < 1:
                #     import ipdb; ipdb.set_trace()
                # assert len(valid_obj_semseg_single_list) >= 1
                if len(valid_obj_semseg_single_list) > 1:
                    if not all([_ in [43, 44, 45] for _ in valid_obj_semseg_single_list]): 
                        print(valid_obj_semseg_single_list), im_info_dict, obj_idx
                        IF_SKIP = True
                        excludes_scene_list.append((im_info_dict['meta_split'], im_info_dict['scene_name'], im_info_dict['frame_id'], obj_idx))
                        print(red('excluded:'), (im_info_dict['meta_split'], im_info_dict['scene_name'], im_info_dict['frame_id'], obj_idx))
                        # import ipdb; ipdb.set_trace()
                    # assert all([_ in [43, 44, 45] for _ in valid_obj_semseg_single_list]) # one object which consists of multiple stuff classes (i.e. "wall_43", "floor_44", "ceiling_45")
                
            if IF_SKIP:
                continue
            
            panoptic_json_annotation = {
                "image_id": image_id,
                "file_name": str(im_pan_seg_rgb_path.relative_to(or_root/'im_pan_seg_rgb')), 
                "image_file_name": str(im_info_dict['im_sdr_path'].relative_to(or_root)), 
                # "file_name": str(im_pan_seg_rgb_path.relative_to(or_root/'im_pan_seg_rgb')), 
                "pan_seg_file_name": str(im_pan_seg_rgb_path.relative_to(or_root/'im_pan_seg_rgb')), 
                "segments_info": segments_info,
            }
            panoptic_json_annotations.append(panoptic_json_annotation)

            panoptic_json_image = {}
            panoptic_json_image["id"] = image_id
            panoptic_json_image["file_name"] = str(im_info_dict['im_sdr_path'].relative_to(or_root))
            panoptic_json_image["width"] = width
            panoptic_json_image["height"] = height
            panoptic_json_images.append(panoptic_json_image)

            # if not im_pan_seg_rgb_path.exists():
            # print('WRITTT ->', str(im_pan_seg_rgb_path))
            im_pan_seg_rgb = panopticapi_utils.id2rgb(im_pan_seg_ids)
            cv2.imwrite(str(im_pan_seg_rgb_path), im_pan_seg_rgb[:, :, ::-1])
            
            if IF_DEBUG:
                for _ in np.unique(im_pan_seg_ids):
                    plt.figure()
                    plt.imshow(im_pan_seg_ids == _)
                    plt.title('id: {}'.format(_))
                    plt.show()

            # else:
            #     print('READDDD: ', str(im_pan_seg_rgb_path))
            #     im_pan_seg_rgb = cv2.imread(str(im_pan_seg_rgb_path))[:, :, ::-1]
            #     pass
            
        # save json
        d = {
            "images": panoptic_json_images,
            "annotations": panoptic_json_annotations,
            # "categories": panoptic_json_categories,
            "categories": OR_46_CATEGORIES,
        }

        save_json(d, out_file)
        print('JSON saved to', out_file)
        
        excludes_scene_list_log_path = os.path.join(dataset_dir, f"OpenRooms_public/im_pan_seg_rgb_{split}_excluded_scenes.txt")
        with open(str(excludes_scene_list_log_path), 'w') as f:
            for _ in excludes_scene_list:
                f.write('%s %s %s %s\n'%(_[0], _[1], _[2], _[3]))
            f.close()
