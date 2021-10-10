import os
import ast
import argparse
import pandas as pd
import time
from tqdm import tqdm 

from ensemble_boxes import *


base_path = './ensemble_models/'
models = os.listdir(base_path)

# 앙상블 모델 확인
print(models)

output_path = base_path + 'output'
os.makedirs(output_path, exist_ok=True)

models_path = [os.path.join(base_path, path) for path in models]

# model number
model_num = 0
for path in models_path:
    if path.split('.')[-1] == 'csv':
        model_num += 1

image_num = pd.read_csv(models_path[0]).shape[0]

test = pd.read_csv(models_path[0])

columns = ['PredictionString']

nms_df = pd.DataFrame(index=range(0,image_num), columns = columns)
nms_df['image_id'] = test['image_id']
softnms_df = pd.DataFrame(index=range(0,image_num), columns = columns)
softnms_df['image_id'] = test['image_id']
non_maximum_weighted_df = pd.DataFrame(index=range(0,image_num), columns = columns)
non_maximum_weighted_df['image_id'] = test['image_id']
weighted_boxes_fusion_df = pd.DataFrame(index=range(0,image_num), columns = columns)
weighted_boxes_fusion_df['image_id'] = test['image_id']

# ensemble_boxes format 
def format_change(image_id):
    # model 단위로 append
    labels_list = []
    scores_list = []
    boxes_list = []

    for model_path in models_path:
        if model_path.split('.')[-1] == 'csv': # csv 파일만 처리

            model = pd.read_csv(model_path)
            prediction = model['PredictionString'][image_id]

            labels = []
            scores = []
            bbox = []

            # type 에러 체크
            if type(model['PredictionString'][image_id]) == str:
                original = prediction.split(' ')
                i = 0
                # label, score, box -> split
                for idx in range(len(model['PredictionString'][image_id])//6):
                    # 마지막 ' ' 예외 처리
                    try:
                        label, score, x,y,w,h = map(float, original[i:i+6])
                        i += 6
                        labels.append(int(label))
                        scores.append(score)
                        bbox.append([x/1024,y/1024,w/1024,h/1024])
                    except: 
                        continue

                labels_list.append(labels)
                scores_list.append(scores)
                boxes_list.append(bbox)
    return labels_list, scores_list, boxes_list

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list " %(s))
    return v

def to_submission_format(boxes, scores, labels):
    output = []
    for idx in range(len(boxes)):
        label = labels[idx]
        score = scores[idx]
        x,y,w,h = boxes[idx]
        output+=[label, score, x*1024, y*1024, w*1024, h*1024]
    output = map(str, output)
    return ' '.join(output)

def nms_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr):

    boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    output = to_submission_format(boxes, scores, labels)
    nms_df['PredictionString'][image_id] = output

def soft_nms_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, sigma, skip_box_thr):
    boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    output = to_submission_format(boxes, scores, labels)
    softnms_df['PredictionString'][image_id] = output

def non_maximum_weighted_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr):
    boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    output = to_submission_format(boxes, scores, labels)
    non_maximum_weighted_df['PredictionString'][image_id] = output

def weighted_boxes_fusion_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr):
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    output = to_submission_format(boxes, scores, labels)
    weighted_boxes_fusion_df['PredictionString'][image_id] = output

    # image 단위 ensemble
def main(args):
    for image_id in tqdm(range(image_num)):

        labels_list, scores_list, boxes_list = format_change(image_id)

        # print(labels_list,scores_list,boxes_list)
        # print(len(labels_list), len(scores_list), len(boxes_list))
        # print(len(labels_list[0]), len(labels_list[1]), len(labels_list[2]))

        # iou_thr = 0.5
        # skip_box_thr = 0.0001
        # sigma = 0.1

        # weights = [2, 1, 1]

        iou_thr = args.iou_thr
        skip_box_thr = args.skip_box_thr
        sigma = args.sigma
        weights = args.weight

        if labels_list and scores_list and boxes_list:

            if args.nms_bool:
                nms_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr)

            if args.soft_nms_bool:
                soft_nms_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, sigma, skip_box_thr)
            
            if args.non_maximum_weighted_bool:
                non_maximum_weighted_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr)

            if args.wbf_bool:
                weighted_boxes_fusion_make(image_id, boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr)

    if args.nms_bool:
        nms_df.to_csv(os.path.join(output_path,'nms.csv'))

    if args.soft_nms_bool:
        softnms_df.to_csv(os.path.join(output_path,'soft_nms.csv'))
        
    if args.non_maximum_weighted_bool:
        non_maximum_weighted_df.to_csv(os.path.join(output_path,'non_maximum_weighted.csv'))

    if args.wbf_bool:
        weighted_boxes_fusion_df.to_csv(os.path.join(output_path,'weighted_boxes_fusion.csv'), index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for creating ensemble csv file')
    parser.add_argument('--iou_thr', type = float, default = 0.5)
    parser.add_argument('--skip_box_thr', default = 0.0001)
    parser.add_argument('--sigma', default = 0.1)

    parser.add_argument('--weight', type = arg_as_list, default = [1]*model_num, help = 'List of info columns')

    parser.add_argument('--nms_bool', default = True)
    parser.add_argument('--soft_nms_bool', default = True)
    parser.add_argument('--non_maximum_weighted_bool', default = True)
    parser.add_argument('--wbf_bool', default = True)

    args = parser.parse_args()

    main(args)