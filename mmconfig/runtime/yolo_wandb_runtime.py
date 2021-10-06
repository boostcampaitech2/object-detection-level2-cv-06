seed = 1004
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='model',
                name='yolov3_d53_fp16_mstrain-608_273e_coco'
            ))
    ])

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]