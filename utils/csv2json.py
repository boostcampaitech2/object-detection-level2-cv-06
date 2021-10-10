import os
import sys
import json
import argparse
import collections

import numpy as np
import pandas as pd
import PIL.Image as Image
import albumentations as A

def csv2df(output_dir: str):
    submission_df = pd.read_csv(output_dir)
    df = pd.DataFrame(columns = ['imgname', 'bbox_id', 'cls', 'conf', 'X', 'Y', 'W', 'H'])
    idcount = 0
    for i in submission_df.shape[0]:
        im_name = submission_df.iloc[i, 1]
        annos = submission_df.iloc[i, 1].split(' ')
        for box_idx in range(len(annos)//6):
            st = box_idx *6
            ed = st + 6
            df.append([im_name, idcount] + annos[st:ed])
            idcount += 1





def read_json(json_dir: str)->pd.DataFrame:
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    source_annotations = source_anns['annotations']
    source_images = source_anns['images']

    img_id = source_images[-1]['id'] + 1
    anno_id = source_annotations[-1]['id'] + 1

    df_anno = pd.json_normalize(source_annotations)
    df_anno[["X","Y","W","H"]] = df_anno['bbox'].tolist()
    df_anno.drop(columns='bbox', inplace=True)

    return source_anns, df_anno, img_id, anno_id


def df2coco(source_anns: dict, processed_anno: pd.DataFrame, anno_id, mosaic_only = False):
    if mosaic_only:
        source_anns['images'] = []
        source_anns['annotations'] = []

    for image_id in processed_anno.image_id.unique():
        image_info = {
                        "width": 1024,
                        "height": 1024,
                        "file_name": "train/" + str(image_id) + '.jpg',
                        "license": 0,
                        "flickr_url": 'null',
                        "coco_url": 'null',
                        "date_captured": 'null',
                        "id": int(image_id)
        }
        source_anns['images'].append(image_info)

    for i in range(processed_anno.shape[0]):
        row = processed_anno.iloc[i, :]
        anno_info = {
                        "image_id": int(row['image_id']),
                        "category_id": row['category_id'],
                        "area": row['area'],
                        "bbox": [row['X'], row['Y'], row['W'], row['H']],
                        "iscrowd": 0,
                        "id": anno_id
        }

        anno_id += 1 
        source_anns['annotations'].append(anno_info)

    return source_anns

