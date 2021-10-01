_base_ = [
    '/opt/ml/detection/mmdetection/configs/_base_/schedules/schedule_1x.py'
]

# optimizer
optimizer = dict(
    _delete_ = True,
    type='Adam', 
    lr=1e-4, 
    weight_decay=0.0001)

# learning policy
lr_config = dict(
    _delete_ = True,
    policy='CosineAnnealing',
    min_lr_ratio=1e-6
)

runner = dict(
    type='EpochBasedRunner', 
    max_epochs=30
)
