_base_ = [
    '../models/detr_r50_8x2_150e_coco_model.py', '../runtime/model_wandb_runtime.py', 
    '../datasets/detr_dataset.py', '../schedules/schedule_adamw_detr_1x.py'
]

# https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth
load_from = '/opt/ml/detection/mmconfig/models/pretrained/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='detr_test'
            ))
    ])
