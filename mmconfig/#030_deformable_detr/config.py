_base_=[
    '/opt/ml/detection/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
    # '/opt/ml/detection/mmdetection/configs/_base_/datasets/coco_detection.py',
    # '/opt/ml/detection/mmdetection/configs/_base_/default_runtime.py',
    # '/opt/ml/detection/mmdetection/configs/_base_/schedules/schedule_1x.py'
]


# --model--
model = dict(
    bbox_head=dict(num_classes=10)
)


# --dataset--
data_root = '/opt/ml/detection/dataset/'
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# train_pipeline = [
#     dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True)
# ]
    
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(img_scale=(1024, 1024))
# ]

data = dict(
    samples_per_gpu = 4,
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes = classes
        ),
    val = dict(
        ann_file=data_root + 'train.json',
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
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',init_kwargs=dict(project='detr',name='def_detr_validate'))
        ]
)


# # --scheduler--
optimizer = dict(
    _delete_=True, 
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        })
)

lr_config = dict(
    _delete_ = True,
    policy='step', 
    step=[20]
)

runner = dict(
    _delete_ = True,
    type='EpochBasedRunner', 
    max_epochs=30
)

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(
        max_norm=0.1, 
        norm_type=2)
)

# --else--
seed = 2021
init_weights = True
work_dir = '/opt/ml/object-detection-level2-cv-06/mmconfig/#030_deformable_detr/work_dir'
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
load_from = '/opt/ml/object-detection-level2-cv-06/mmconfig/#030_deformable_detr/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
#pth: https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth