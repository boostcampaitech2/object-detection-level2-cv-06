_base_ = [
    '/opt/ml/detection/mmdetection/configs/_base_/default_runtime.py'
]

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',init_kwargs=dict(project='detr',name='faster_rcnn'))
        ]
    )


