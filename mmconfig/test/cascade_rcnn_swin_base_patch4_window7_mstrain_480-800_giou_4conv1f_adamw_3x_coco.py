_base_ = [
    '../models/cascade_rcnn_swin_fpn_base.py',
    '../datasets/valid_search_datset_swin-t.py',
    '../schedules/schedule_adamw_3x.py', '../runtime/valid_search_wandb_runtime.py'
]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='cascade_rcnn_swin_base_pretrained'
            ))
    ])

# Mixed Precision Training
fp16 = dict(loss_scale='dynamic')
