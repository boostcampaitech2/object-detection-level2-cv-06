_base_ = '../models/yolov3_d53_mstrain-608_273e_coco.py'

# init_weights = True
load_from = load_from = '/opt/ml/detection/mmconfig/models/pretrained/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'

# fp16 settings
fp16 = dict(loss_scale='dynamic')
