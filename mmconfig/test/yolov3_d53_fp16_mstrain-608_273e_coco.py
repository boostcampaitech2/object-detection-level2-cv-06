_base_ = ['../models/yolov3_d53_mstrain-608_273e_coco.py', '../runtime/model_wandb_runtime.py', 
          '../datasets/yolov3_dataset.py', '../schedules/schedule_yolo_grad_clip_1x.py']

# https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_fp16_mstrain-608_273e_coco/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth
load_from = '/opt/ml/detection/mmconfig/models/pretrained/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'

# fp16 settings
fp16 = dict(loss_scale='dynamic')


log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='yolov3_test'
            ))
    ])

