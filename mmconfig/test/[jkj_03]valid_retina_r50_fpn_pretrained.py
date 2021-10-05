_base_ = [
    '../models/retinanet_r50_fpn_class10.py',
    '../datasets/valid_search_dataset.py',
    '../schedules/schedule_adam_2x.py', '../runtime/valid_search_wandb_runtime.py'
]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

data_root = '/opt/ml/detection/dataset/'
data = dict(
    train=dict(ann_file=data_root + 'candidate/' + 'jkj_03_train.json',),
    val=dict(ann_file=data_root + 'candidate/' + 'jkj_03_valid.json',),
    test=dict(ann_file=data_root + 'test.json',)
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='valid_search',
                name='[jkj_03]valid_retinanet_r50_fpn_pretrained' # ex) [jkj_01]valid_faster-rcnn_pretrained
            ))
    ])


load_from='/opt/ml/detection/mmconfig/models/pretrained/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
# path: https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
