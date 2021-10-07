_base_=[
    '../datasets/valid_search_dataset_detr.py',
    '../models/deformable_detr_r50_16x2_50e_coco_class10.py',
    '../schedules/schedule_adamw_1x.py', 
    '../runtime/valid_search_wandb_runtime.py'
]

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

# init_weights = True
load_from = '/opt/ml/detection/mmconfig/models/pretrained/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'

data_root = '/opt/ml/detection/dataset/'
data = dict(
    samples_per_gpu=4,
    train=dict(ann_file=data_root + 'candidate/' + 'jkj_03_train.json',),
    val=dict(ann_file=data_root + 'candidate/' + 'jkj_03_valid.json',),
    test=dict(ann_file=data_root + 'test.json',)
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detr',
                name='2stage_def_detr_ptr' # ex) [jkj_01]valid_faster-rcnn_pretrained
            ))
    ])

#path: https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth