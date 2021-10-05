_base_ = [
    '../../mmdetection/configs/_base_/models/retinanet_r50_fpn.py'
]

model = dict(
    bbox_head=dict(
        num_classes=10,
    )
)

# path: https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
