_base_ = [
    '../models/cascade_rcnn_r50_fpn_class10.py',
    #'coco_instance.py',
    '../datasets/team_base_dataset_aug.py',
    '../runtime/valid_search_wandb_runtime.py',
    '../schedules/schedule_2x.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768])
        #[256, 512, 1024, 2048]
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    # crop
    dict( type='RandomSizedBBoxSafeCrop', width=1024, height=1024, erosion_rate=0.3, p=0.3),
    # flip
    dict(
        type='OneOf',
        transforms=[
            # dict(type='HorizontalFlip', p=1.0),
            # dict(type='VerticalFlip', p=1.0),
            dict(type='Affine', p=1.0, shear=15),
            # dict(type='ShiftScaleRotate', p=1.0)
    ], p=0.1),

    # color
    dict(
        type='OneOf',
        transforms=[
            dict(type='RGBShift', r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            dict(type='ToGray', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='RandomBrightnessContrast', p=1.0),
        ], p=0.1),
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

    dict(type='Emboss', p=0.2, alpha=[0.4, 0.6], strength=[0.3, 0.7])
]

# augmentation strategy originates from DETR / Sparse RCNN
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
            'pad_shape', 'scale_factor')),
]
val_pipeline = [
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline),val=dict(pipeline=val_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')

# checkpoint_config = dict(interval=1)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='valid_search',
        #         name='YOUR_EXP'
        #     ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=30)
