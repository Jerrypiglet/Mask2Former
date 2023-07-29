'''
or-matseg-sem

- data location: /data/OpenRooms_public/im_matseg_sem_pan_seg_rgb; /data/OpenRooms_public/im_matseg_sem_pan_seg_rgb_{train, val}.json
- register in Detectron2 as: or_matseg_sem_panoptic_{train, val}

OpenRooms material parts labels, where:

- each material part which belongs to an object is a 'thing', with semantic label of the object category;
- each material part which belongs to the background (wall, ceiling, floor) is a 'stuff', with semantic label of wall/ceiling/floor.
'''

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
im_matseg_sem_pan_seg_rgb_root = Path('/data/OpenRooms_public/im_matseg_sem_pan_seg_rgb')
frame_list_root = project_root_path / 'data/openrooms/public'

from prepare_or_pan_seg import OR_SEM_SEG_CATEGORIES, PALETTE, get_im_info_list, get_exclude_scenes

# DEBUG = True; debug_frame_id = 95
DEBUG = False

if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

    # for split in ['train', 'val']:
    # for split in ['val']:
    for split in ['train']:
        
        image_dir = os.path.join(dataset_dir, f"OpenRooms_public")

        # folder to store panoptic PNGs
        out_folder = os.path.join(dataset_dir, f"OpenRooms_public/im_matseg_sem_pan_seg_rgb/")
        # json with segmentations information
        out_file = os.path.join(dataset_dir, f"OpenRooms_public/im_matseg_sem_pan_seg_rgb_{split}.json")

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
        
        excludes_scene_list = get_exclude_scenes(frame_list_root / 'exclude_semseg.txt')
        im_info_list = [_ for _ in im_info_list if (_['meta_split_semantics'], _['scene_name']) not in excludes_scene_list]
        
        if DEBUG:
            im_info_list = im_info_list[debug_frame_id:debug_frame_id+1] # DEBUG
        
        # process matseg
        for image_id, im_info_dict in tqdm(enumerate(im_info_list)):
            panoptic_json_annotation = {}
            
            filename = str(im_info_dict['im_sdr_path'])
            height, width = load_img(im_info_dict['im_sdr_path']).shape[:2]
            # print(height, width)

            im_matseg_sem_pan_seg_rgb_folder = im_matseg_sem_pan_seg_rgb_root / (im_info_dict['meta_split'].replace('DiffMat', '').replace('DiffLight', '')) / im_info_dict['scene_name']
            im_matseg_sem_pan_seg_rgb_folder.mkdir(parents=True, exist_ok=True)
            im_matseg_sem_pan_seg_rgb_path = im_matseg_sem_pan_seg_rgb_folder / ('im_matseg_sem_pan_seg_rgb_%s.png'%im_info_dict['frame_id'])
            
            # if im_matseg_sem_pan_seg_rgb_path.exists():
            #     continue
            
            # load objs from matseg labels
            print('===', image_id, im_info_dict['im_sdr_path'])
            
            im_matseg_dict = load_matseg(im_info_dict['matseg_path'])
            if im_matseg_dict is None:
                print(red('im_matseg_dict is None (something wrong with the matseg label: %s)'%str(im_info_dict['matseg_path'])))
                continue
            
            mat_aggre_map = im_matseg_dict['mat_aggre_map']
            num_mat_masks = im_matseg_dict['num_mat_masks']

            # load semseg labels
            im_semseg = load_img(im_info_dict['semseg_path'], ext='npy').astype('uint8')
            im_semseg = im_semseg - 1
            im_semseg[im_semseg == -1] = 255
            
            # load object masks
            im_mask = load_img(im_info_dict['immask_path'], ext='png')[:, :, 0] / 255. # [0., 1.]
            seg_obj = im_mask > 0.9
            
            im_matseg_sem_pan_seg_ids = np.zeros((height, width), dtype=np.int16)
            segments_info = []
            
            mask_count = 1 # 0 is for background
    
            IF_SKIP = False
            
            for mask_id in np.arange(1, num_mat_masks+1):
                # for num_mat_mask in np.arange(1, 2):
                mask_single = mat_aggre_map == mask_id
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(mask_single)
                # plt.subplot(1, 2, 2)
                # plt.imshow(seg_obj)
                # plt.show()
                mask_single[seg_obj == False] = False
                # print(np.sum(mask_single))
                # if np.sum(mask_single) < 500:
                #     continue
                
                # assert np.amax(mask_id) < 256
                area = np.sum(mask_single)  # segment area computation
                if area < 500: continue # skip segments with very small area
                
                im_matseg_sem_pan_seg_ids[mask_single] = mask_count
                semseg_masked = im_semseg[mask_single]
                semseg_counts = [np.sum(semseg_masked == i) for i in np.unique(semseg_masked)]
                semseg_counts_sort_idx = np.argsort(np.array(semseg_counts))[::-1]
                semseg_id = int(np.unique(semseg_masked)[semseg_counts_sort_idx[0]])
                
                is_thing = OR_SEM_SEG_CATEGORIES[semseg_id] not in ["wall_43", "floor_44", "ceiling_45"]
                
                print(image_id, mask_count, semseg_masked.shape, np.unique(semseg_masked), [np.sum(semseg_masked == i) for i in np.unique(semseg_masked)])
                print(float(np.amax(np.array(semseg_counts))) / semseg_masked.shape[0], semseg_id, OR_SEM_SEG_CATEGORIES[semseg_id], 'total area:', area)

                # if not DEBUG:
                #     assert float(np.amax(np.array(semseg_counts))) / semseg_masked.shape[0] > 0.7
                if float(np.amax(np.array(semseg_counts))) / semseg_masked.shape[0] < 0.8:
                    IF_SKIP = True
                
                segments_info.append({
                    'id': int(mask_count),
                    'category_id': semseg_id,
                    'iscrowd': 0,
                    # 'isthing': 0,
                    # 'isstuff': 1,
                    'isthing': int(is_thing),
                    'isstuff': int(not is_thing), 
                    'area': int(area),
                })
                
                mask_count += 1
            '''
            check to make sure there is only one material seg as stuff, for each of ["wall_43", "floor_44", "ceiling_45"]
            '''
            for _ in ["wall_43", "floor_44", "ceiling_45"]:
                _stuff_counts = [1 for seg_info in segments_info if seg_info['isthing'] == 0 and OR_SEM_SEG_CATEGORIES[seg_info['category_id']] == _]
                # if not DEBUG:
                #     assert len(_stuff_counts) in [1, 0], 'more than one stuff (%d segments) for %s'%(len(_stuff_counts), _)
                if len(_stuff_counts) not in [1, 0]:
                    IF_SKIP = True
                    
            if IF_SKIP:
                excludes_scene_list.append((im_info_dict['meta_split'], im_info_dict['scene_name'], im_info_dict['frame_id']))
                print(red('excluded:'), (im_info_dict['meta_split'], im_info_dict['scene_name'], im_info_dict['frame_id']))
                continue

                
            '''
            write labels to file
            '''
                
            panoptic_json_annotation = {
                "image_id": image_id,
                "file_name": str(im_matseg_sem_pan_seg_rgb_path.relative_to(or_root/'im_matseg_sem_pan_seg_rgb')), 
                "image_file_name": str(im_info_dict['im_sdr_path'].relative_to(or_root)), 
                # "file_name": str(im_matseg_sem_pan_seg_rgb_path.relative_to(or_root/'im_matseg_sem_pan_seg_rgb')), 
                "pan_seg_file_name": str(im_matseg_sem_pan_seg_rgb_path.relative_to(or_root/'im_matseg_sem_pan_seg_rgb')), 
                "segments_info": segments_info,
            }
            panoptic_json_annotations.append(panoptic_json_annotation)

            panoptic_json_image = {}
            panoptic_json_image["id"] = image_id
            panoptic_json_image["file_name"] = str(im_info_dict['im_sdr_path'].relative_to(or_root))
            panoptic_json_image["width"] = width
            panoptic_json_image["height"] = height
            panoptic_json_images.append(panoptic_json_image)

            # if not im_matseg_sem_pan_seg_rgb_path.exists():
            # print('WRITTT ->', str(im_matseg_sem_pan_seg_rgb_path))
            im_matseg_sem_pan_seg_rgb = panopticapi_utils.id2rgb(im_matseg_sem_pan_seg_ids)
            cv2.imwrite(str(im_matseg_sem_pan_seg_rgb_path), im_matseg_sem_pan_seg_rgb[:, :, ::-1])
            # else:
            #     print('READDDD: ', str(im_matseg_sem_pan_seg_rgb_path))
            #     im_matseg_sem_pan_seg_rgb = cv2.imread(str(im_matseg_sem_pan_seg_rgb_path))[::-1]
            #     pass
            
        # save json
        d = {
            "images": panoptic_json_images,
            "annotations": panoptic_json_annotations,
            "categories": panoptic_json_categories,
        }

        save_json(d, out_file)
        print('JSON faved to', out_file)
        
        excludes_scene_list_log_path = os.path.join(dataset_dir, f"OpenRooms_public/im_matseg_sem_pan_seg_rgb_{split}_excluded_scenes.txt")
        with open(str(excludes_scene_list_log_path), 'w') as f:
            for _ in excludes_scene_list:
                f.write('%s %s %s %s\n'%(_[0], _[1], _[2]))
            f.close()

