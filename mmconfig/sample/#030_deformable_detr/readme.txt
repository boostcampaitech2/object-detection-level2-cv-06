##학습 전 pretrained weight를 받아주세요
    https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth

#finetuning한 weight는 슬랙에 올려두겠습니다
    
##학습 전 wandb 설정을 해주세요
##학습 전 경로설정을 수동으로 해주세요
    config.py --> work_dir
    config.py --> load_from
#train
conda activate detection
cd /opt/ml/detection/mmdetection
python tools/train.py <config.py 경로>

#inference
conda activate detection
python <inference.py 경로>
