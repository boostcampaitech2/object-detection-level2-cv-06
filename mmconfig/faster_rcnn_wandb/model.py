_base_ = ['/opt/ml/detection/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py']

# bbox_head = [
#     dict(num_classes = 10),
#     dict(num_classes = 10),
#     dict(num_classes = 10),
# ]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)
        )
)
