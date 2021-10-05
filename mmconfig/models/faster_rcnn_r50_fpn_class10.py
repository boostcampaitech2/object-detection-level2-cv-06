_base_ = [
    '../../mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
]
# /opt/ml/detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10,
        )
    )
)

#pth : https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth