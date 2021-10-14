_base_ = [
    '../models/universenet101_gfl.py',
    '../datasets/team_base_dataset_aug.py',
    '../schedules/schedule_2x.py', '../runtime/valid_search_wandb_runtime.py'
]

model = dict(
    bbox_head=dict(num_classes=10))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=30)
# # --runtime--
# log_config = dict(
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='WandbLoggerHook',init_kwargs=dict(project='model',name='universenet'))
#         ]
# )

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='[aug 04] new_aug_universenet_pretrained' # ex) [jkj_01]valid_faster-rcnn_pretrained
            ))
    ])


# load_from ='https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth'
load_from = '/opt/ml/detection/mmconfig/models/pretrained/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth'