_base_ = [
    '../../configs/_base_/default_runtime.py'
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='valid_search',
                name='YOUR_EXP'
            ))
    ])