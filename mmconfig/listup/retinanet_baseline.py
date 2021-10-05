_base_ = [
    '../models/retinanet_r50_fpn.py',
    '../datasets/valid_search_dataset.py',
    '../schedules/schedule_adam_2x.py', '../runtime/valid_search_wandb_runtime.py'
]

data_root = '/opt/ml/detection/dataset/'
data = dict(
    train=dict(ann_file=data_root + 'candidate/' + 'ljh_01_train.json',),
    val=dict(ann_file=data_root + 'candidate/' + 'ljh_01_valid.json',),
    test=dict(ann_file=data_root + 'test.json',)
)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='valid_search',
                name='retina'
            ))
    ])

load_from='/opt/ml/detection/mmdetection/weights/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
# path: https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
