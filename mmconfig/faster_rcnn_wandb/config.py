_base_=[
    'model.py',
    'dataset.py',
    'wandb_runtime.py',
    'CAscheduler.py'
]

seed = 1004
work_dir = '/opt/ml/personel/swin/work_dir'    #반드시 절대경로여야 함
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
# load_from = '/opt/ml/detection/mmdetection/pretrained/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'