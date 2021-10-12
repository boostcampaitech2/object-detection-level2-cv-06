_base_ = [
    '../datasets/team_base_dataset_aug_3.py',
    '../models/faster_rcnn_r50_fpn_class10.py',
    '../schedules/schedule_adam_2x.py', 
    '../runtime/model_wandb_runtime.py'
]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# init_weights = True
load_from = '/opt/ml/detection/mmconfig/models/pretrained/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

# data_root = '/opt/ml/detection/dataset/'
# data = dict(
#     train=dict(ann_file=data_root + 'candidate/' + 'jkj_01_train.json',),
#     val=dict(ann_file=data_root + 'candidate/' + 'jkj_01_valid.json',),
#     test=dict(ann_file=data_root + 'test.json',)
# )

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='[aug_03] new_aug_faster-rcnn_pretrained' # ex) [jkj_01]valid_faster-rcnn_pretrained
            ))
    ])

runner = dict(type='EpochBasedRunner', max_epochs=30)

seed=1004

