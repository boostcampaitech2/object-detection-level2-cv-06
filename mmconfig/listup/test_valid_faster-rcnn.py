_base_ = [
    '../../mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../mmdetection/configs/_base_/schedules/schedule_2x.py', 
    '../datasets/valid_search_dataset.py',
    '../runtime/valid_search_wandb_runtime.py'
]

data_root = '/opt/ml/detection/dataset/'
data = dict(
    train=dict(ann_file=data_root + 'candidate/' + 'train_vaild_split_sample_train.json',),
    val=dict(ann_file=data_root + 'candidate/' + 'train_vaild_split_sample_valid.json',),
    test=dict(ann_file=data_root + 'test.json',)
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='valid_search',
                name='test_valid_faster-rcnn'
            ))
    ])

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10,
        )
    )
)