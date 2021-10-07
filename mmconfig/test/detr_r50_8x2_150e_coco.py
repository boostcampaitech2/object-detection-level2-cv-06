_base_ = [
    '../models/detr_r50_8x2_150e_coco_model.py', '../runtime/valid_search_wandb_runtime.py', 
    '../datasets/detr_dataset.py', '../schedules/schedule_adamw_detr_1x.py'
]

# init_weights = True
load_from = '/opt/ml/detection/mmconfig/models/pretrained/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

