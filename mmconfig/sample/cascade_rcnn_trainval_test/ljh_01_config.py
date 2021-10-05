_base_=[
    '/opt/ml/detection/mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '/opt/ml/detection/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/opt/ml/detection/mmdetection/configs/_base_/default_runtime.py',
    '/opt/ml/detection/mmdetection/configs/_base_/schedules/schedule_1x.py'
]


# --model--
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    )
)


# --dataset--
data_root = '/opt/ml/detection/dataset/'
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

train_pipeline = [
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True)
]
    
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(img_scale=(512, 512))
]

data = dict(
    samples_per_gpu = 16,
    train=dict(
        ann_file=data_root + 'candidate/train_vaild_split_sample_train.json',
        img_prefix=data_root,
        classes = classes
        ),
    val = dict(
        ann_file=data_root + 'candidate/train_vaild_split_sample_valid.json',
        img_prefix=data_root,
        classes = classes
    ),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes
        )
)


# --runtime--
log_config = dict(
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',init_kwargs=dict(project='detr',name='cascade_rcnn'))
        ]
)


# --scheduler--
optimizer = dict(
    _delete_ = True,
    type='Adam', 
    lr=1e-4, 
    weight_decay=0.0001
)
lr_config = dict(
    _delete_ = True,
    policy='step',
    step = [20]
)
runner = dict(
    type='EpochBasedRunner', 
    max_epochs=30
)


# --else--
seed = 1004
work_dir = '/opt/ml/personel/cascade_rcnn/work_dir'    #반드시 절대경로여야 함
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
load_from = '/opt/ml/detection/mmdetection/pretrained/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
