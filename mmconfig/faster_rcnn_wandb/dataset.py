_base_=[
    '/opt/ml/detection/mmdetection/configs/_base_/datasets/coco_detection.py',
]

data_root = '/opt/ml/detection/dataset/'
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

train_pipeline = [
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True)
]
    

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(img_scale=(1024, 1024),)
]

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




