_base_=[
    '../datasets/valid_search_dataset.py',
    '../models/cascade_rcnn_r50_fpn_class10.py',
    '../schedules/schedule_adam_2x.py', 
    '../runtime/valid_search_wandb_runtime.py'
]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# init_weights = True
load_from = '/opt/ml/detection/mmconfig/models/pretrained/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

data_root = '/opt/ml/detection/dataset/'
data = dict(
    train=dict(ann_file=data_root + 'candidate/' + 'jkj_01_train.json',),
    val=dict(ann_file=data_root + 'candidate/' + 'jkj_01_valid.json',),
    test=dict(ann_file=data_root + 'test.json',)
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='valid_search',
                name='[jkj_01]valid_cascade_rcnn_r50_fpn_pretrained' # ex) [jkj_01]valid_faster-rcnn_pretrained
            ))
    ])