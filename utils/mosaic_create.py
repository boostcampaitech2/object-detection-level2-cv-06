import os
import sys
import json
import argparse
import collections

import numpy as np
import pandas as pd
import PIL.Image as Image
import albumentations as A



def json2df(anno_source: str)->pd.DataFrame:
    with open(anno_source) as json_file:
        source_anns = json.load(json_file)

    source_annotations = source_anns['annotations']
    source_images = source_anns['images']

    img_id = source_images[-1]['id'] + 1
    anno_id = source_annotations[-1]['id'] + 1

    df_anno = pd.json_normalize(source_annotations)
    df_anno[["X","Y","W","H"]] = df_anno['bbox'].tolist()
    df_anno.drop(columns='bbox', inplace=True)

    return source_anns, df_anno, img_id, anno_id


def resize_pipe(image, df_anno: pd.DataFrame):
    #bboxes: XYWH format
    bboxes = df_anno[['X', 'Y', 'W', 'H', 'category_id']].values.tolist()
    transform = A.Compose(
        [A.Resize(height=1024, width=1024)], 
        bbox_params=A.BboxParams(format='coco')
        )

    transformed = transform(image=image, bboxes=bboxes)
    re_image, re_anno = transformed['image'], transformed['bboxes']

    for i in range(df_anno.shape[0]):
        df_anno.loc[i, 'X'] = re_anno[i][0]
        df_anno.loc[i, 'Y'] = re_anno[i][1]
        df_anno.loc[i, 'W'] = re_anno[i][2]
        df_anno.loc[i, 'H'] = re_anno[i][3]

    return re_image, df_anno
    

def get_image_id(df_anno: pd.DataFrame, image_num: int, bbox_number = 0) -> list:
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


def concat_images_bbox(df_anno: pd.DataFrame, image_dir: str, image_ids: list, new_id: int) -> np.ndarray:
    #input: origin image(1024*1024)
    assert len(image_ids) == 4
    from PIL import Image
    id_list  = [str(i).zfill(4) for i in image_ids]
    stack = [Image.open(os.path.join(image_dir, f"{id}.jpg")) for id in id_list]
    
    concated_image = np.concatenate(
        (
            np.concatenate((stack[0], stack[1]), axis=1), 
            np.concatenate((stack[2], stack[3]), axis=1)
         ), 
        axis=0
    )

    assert concated_image.shape == (2048, 2048, 3)

    quarter_dict = {0: (0, 0), 1: (1024, 0), 2: (0, 1024), 3: (1024, 1024)}
    nid = new_id
    df_stack = pd.DataFrame(columns = df_anno.columns)
    box_count = 0
    for quarter, id in enumerate(image_ids):
        concat_anno = df_anno[df_anno['image_id'] == id].copy()
        box_count += concat_anno.shape[0]
        concat_anno['image_id'] = nid
        concat_anno['X'], concat_anno["Y"] = concat_anno['X'] + quarter_dict[quarter][0], concat_anno["Y"] + quarter_dict[quarter][1]

        df_stack = pd.concat([df_stack, concat_anno], axis = 0, ignore_index = True)
    assert df_stack.shape[0] == box_count

    return concated_image, df_stack, nid


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


def main(args):
    image_num = int(args.num)
    anno_source = args.dir #'/opt/ml/detection/dataset/train.json'
    # by_image = True
    file_name = 'add_mosaic'
    data_dir = os.path.abspath(os.path.join(anno_source, os.pardir)) + '/train/'
    bbox_constraint = int(args.bbox_num)
    mosaic_only = bool(int(args.mosaic_only))

    custom_json_dir = '/opt/ml/detection/dataset/candidate'
    extension = '.json'

    source_anns, df_anno, img_id_s, anno_id_s = json2df(anno_source)
    images = get_image_id(df_anno, image_num, bbox_number=bbox_constraint)

    stack = []
    nid_stack = []
    processed_anno = pd.DataFrame(columns = df_anno.columns)

    for image in images:
        stack.append(image)

        if len(stack) == 4:
            concat_image, concat_anno, nid = concat_images_bbox(df_anno, data_dir, stack, img_id_s)
            nid_stack.append(nid)
            resized_image, resized_anno = resize_pipe(concat_image, concat_anno)

            Image.fromarray(resized_image).save(os.path.join(data_dir, str(nid))+'.jpg', format=None)
            processed_anno = processed_anno.append(resized_anno)

            img_id_s += 1
            stack = []


    # save processed_anno
    final_json = df2coco(source_anns, processed_anno, anno_id_s, mosaic_only)
    output_path = os.path.join(custom_json_dir, file_name+'_train_' + f'mosaiconly_{str(mosaic_only)}' + extension)   

    with open(output_path, 'w') as outfile:
        json.dump(final_json, outfile, indent=2)

    print(f"file saved : {output_path}")
    print(f'img_saved : {data_dir}, {nid_stack[0]} ~ {nid_stack[-1]}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script for creating image mosaic input: origin json(COCO), output: mosaic-added json(COCO)')
    #how to add optional variables?
    parser.add_argument('--dir', default = '/opt/ml/detection/dataset/team_train.json', help='direction of json train file(absolute path)')
    # parser.add_argument('--start_from', default = 4883, help='start of file id')
    parser.add_argument('--bbox_num', default = 0, help='if not 0, image under bbox number of N will selected. else all')
    parser.add_argument('--mosaic_only', default = 0, help = 'if 1, mosaic_only')
    parser.add_argument('--num', default = 5, help='number of mosaic image warning: image may duplicate if mosaic num > 4 * image num(with or without constraint). use at your own risk')
    # parser.add_argument('--mode', help='if under_num, ')
    # json must be under dataset dir
    args = parser.parse_args()

    main(args)
