_base_ = [
    '../models/universenet101_gfl.py',
    '../datasets/coco_detection_mstrain_480_960.py',
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

# # --runtime--
# log_config = dict(
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='WandbLoggerHook',init_kwargs=dict(project='model',name='universenet'))
#         ]
# )

load_from ='https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth'