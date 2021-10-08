_base_ = [
    '/opt/ml/detection/mmdetection/configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py',
    '/opt/ml/detection/mmconfig/datasets/cbnet_dataset.py',
    '/opt/ml/detection/mmconfig/schedules/schedule_1x.py',
    '/opt/ml/detection/mmconfig/runtime/valid_search_wandb_runtime.py'
]


# do not use mmdet version fp16
runner = dict(type='EpochBasedRunnerAmp', max_epochs=20)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

load_from = '/opt/ml/detection/mmconfig/models/pretrained/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco.pth'
#path: https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_giou_4conv1f_coco.pth.zip