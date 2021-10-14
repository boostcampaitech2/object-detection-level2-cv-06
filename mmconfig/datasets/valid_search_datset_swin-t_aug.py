# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



albu_train_transforms = [
    # crop
    # dict( type='RandomSizedBBoxSafeCrop', width=1024, height=1024, erosion_rate=0.3, p=0.3),
    # flip
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='HorizontalFlip', p=1.0),
    #         dict(type='VerticalFlip', p=1.0),
    #         dict(type='Affine', p=1.0, shear=15),
    #         dict(type='ShiftScaleRotate', p=1.0)
    # ], p=0.2),

    # color
    dict(
        type='OneOf',
        transforms=[
            dict(type='RGBShift', r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            dict(type='ToGray', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='RandomBrightnessContrast', p=1.0),
        ], p=0.3),
    # blur
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ], p=0.2),
    # texture
    dict(
        type='OneOf',
        transforms=[
            dict(type='CLAHE', p=1.0,  clip_limit=5), 
            dict(type='Sharpen', p=1.0)
    ], p=0.2),

    dict(type='Emboss', p=0.4, alpha=[0.4, 0.6], strength=[0.3, 0.7])
]


# augmentation strategy originates from DETR, except for size_divisor=32
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                           (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                           (736, 1024), (768, 1024), (800, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1024), (500, 1024), (600, 1024)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1024), (512, 1024), (544, 1024),
                                     (576, 1024), (608, 1024), (640, 1024),
                                     (672, 1024), (704, 1024), (736, 1024),
                                     (768, 1024), (800, 1024)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc', # coco
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'team_valid.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
