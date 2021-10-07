import argparse
import os
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import collections


def json2df(anno_source: str)->pd.DataFrame:
    with open(anno_source) as json_file:
        source_anns = json.load(json_file)

    source_info = source_anns['info']
    source_licenses = source_anns['licenses']
    source_images = source_anns['images']
    source_categories = source_anns['categories']
    source_annotations = source_anns['annotations']

    df_anno = pd.json_normalize(source_annotations)
    df_anno[["X","Y","W","H"]] = df_anno['bbox'].tolist()
    df_anno.drop(columns='bbox', inplace=True)

    return df_anno


def get_image_id(df_anno: pd.DataFrame, image_num: int, bbox_number) -> list:
    if bbox_number != 0:
        image_list = []
        count = collections.Counter(df_anno['image_id'].to_list())
        for id, val in count.items():
            if val < bbox_number:
                image_list.append(id)
        return np.random.choice(image_list, image_num * 4)

    else:
        image_list = df_anno['image_id'].unique()
        return np.random.choice(image_list, image_num * 4)


def scale_boxes(df_anno: pd.DataFrame, images: list):
    assert len(images) == 4
    
    for i in images:
        df_id = df_anno[df_anno['image_id'] == i]
    return None


def add_images(images: list) -> np.ndarray:
    assert len(images) == 4
    a, b, c, d = images
    ab = np.concatenate((a, b), axis=1)
    cd = np.concatenate((c, d), axis=1)
    concated_image = np.concatenate((ab, cd), axis=0)

    return concated_image


def main(args):
    image_num = args.num
    anno_source = args.dir #'/opt/ml/detection/dataset/train.json'
    by_image = True
    file_name = 'add_mosaic'
    data_dir = os.path.abspath(os.path.join(anno_source, os.pardir))


    custom_json_dir = '/opt/ml/detection/dataset/candidate'
    extension = '.json'

    df_anno = json2df(anno_source)
    images = get_image_id(df_anno)
    stack = []

    for image in images:
        if len(stack) == 4:
            


        else:
            stack.append(image)


if __name__=='main':
    parser = argparse.ArgumentParser(description='script for creating image mosaic input: origin json(COCO), output: mosaic-added json(COCO)')
    #how to add optional variables?
    parser.add_argument('--dir', help='direction of json train file(absolute path)')
    parser.add_argument('--start_from', default = 4883, help='start of file id')
    parser.add_argument('--bbox_num', default = 0, help='if not 0, image under bbox number of N will selected. else all')
    parser.add_argument('--num', help='number of mosaic image warning: image may duplicate if nosaic num > 4 * image num(with or without constraint). use at your own risk')
    # parser.add_argument('--mode', help='if under_num, ')
    # json must be under dataset dir
    args = parser.parse_args()

    main(args)