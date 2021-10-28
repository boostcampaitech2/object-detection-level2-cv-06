# 칠성사이다
![image](https://user-images.githubusercontent.com/20790778/137433985-622be56d-82eb-4dd7-bbec-c7079b0bf059.png)

| [강지우](https://github.com/jiwoo0212) | [곽지윤](https://github.com/kwakjeeyoon) | [서지유](https://github.com/JiyouSeo) | [송나은](https://github.com/sne12345) | [오재환](https://github.com/jaehwan-AI) | [이준혁](https://github.com/kmouleejunhyuk) | [전경재](https://github.com/ppskj178) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/68782183/138297784-223d2d61-74f7-4a19-8aaf-5525309e2bd8.jpg) | ![image](https://user-images.githubusercontent.com/55044675/138575690-216fc641-dba1-4737-a571-6b1058e780b2.jpg) | ![image](https://avatars.githubusercontent.com/u/61641072?v=4) | ![image](https://user-images.githubusercontent.com/68782183/138638320-19b24d42-6014-4042-b443-cbeb50251cfd.jpg) | ![image](https://user-images.githubusercontent.com/68782183/138295480-ca0169cd-5c40-44ae-b222-d74d9cc4bc82.jpg) | ![d](https://user-images.githubusercontent.com/49234207/138424590-385b34c2-fae2-426f-8abe-8b40d21ba766.jpg)| ![image](https://user-images.githubusercontent.com/20790778/138396418-b669cbed-40b0-45eb-9f60-7167cae739b7.png) | |


## Competition Overview
- 목적: 사진에서 쓰레기를 Detection 하는 모델 제작
- Dataset: 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋 9700여장(train 50%, 대회 평가 50%)
- 평가 metric: mean Average Precision(mAP50) on test dataset


## Requirements
- python, numpy, pandas
- pytorch
- torchvision
- albumentations
- seaborn
- wandb
- [UniverseNet MMdetection](https://github.com/shinya7y/UniverseNet) [include [Original MMdetection](https://github.com/open-mmlab/mmdetection)]
- [Bbox Ensemble library](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
> 본 프로젝트는 MMdetection 라이브러리를 포함하는 UniverseNet MMdetection을 사용하였습니다.<br>
> [UniverseNet MMdetection](https://github.com/shinya7y/UniverseNet)을 clone하여 `mmdetection`이라는 디렉토리명으로 아래 폴더 구조에 맞게 위치시켜 주시면 됩니다.


## contents
```
object-detection-level2-cv-06(*detection)/
│
├── EDA/ # custom EDA files
│
├── dataset/ # 전체 데이터 셋
│   ├─ candidate/ # custom dataset candidate
│   ├─ sample.json # just sample
│   ├─ test.json # orginal test.json, full data
│   ├─ train.json # orginal train.json, full data
│   ├─ team_train.json # 팀에서 자체적으로 설정한 train set (train.json의 subset)
│   └─ team_valid.json # 팀에서 자체적으로 설정한 valid set (train.json의 subset)
│
├── mmconfig/ # mmdetection용 config 모음
│   ├─ datasets/ # dataset configs
│   ├─ models/ # model configs
│   │  └─ pretrained/ # pretrained save pth path
│   ├─ schedules/ # schedule configs
│   ├─ runtime/ # runtime configs
│   ├─ sample/ # old style configs, current not used
│   ├─ listup/ # test config에서 선택되어 분류된 config들
│   └─ test/ # [datasets, models, schedules, runtime] config 들을 통합한 main config
│
├── utils/ # 부가적으로 필요한 util들  
│   ├─ ensemble_models/ 앙상블 할 출력물 경로(해당 폴더내에 csv파일을 대상으로 선택한다)
│   │  └─ output/ 앙상블 결과 파일이 저장되는 경로
│   ├─ EDA.ipynb # main EDA
│   ├─ anno_converter.ipynb # coco dataset json 파일을 조건에따라 샘플링하여 파일로 저장해주는 converter
│   ├─ csv2json.py # 제출용 submission csv 파일을 다시 coco 형태의 json 파일로 변경해주는 파일 (presudo labeling 등에 사용)
│   ├─ final_ensemble.py # ensemble_models/안의 csv 파일을 여러 옵션으로 앙상블하여 output/에 저장
│   ├─ inference.py # mmdetection 에서 생성된 output 모델을 inference 하여 결과 csv 파일 생성
│   ├─ merge_json.py # 여러 coco 형태의 json 파일 하나의 파일로 병합 (단, mmdetection 에서는 dataset을 list 형태로 여러개 넣어주는 방법도 가능하다)
│   ├─ mosaic_create.py # dataset의 image와 annotation을 모자이크 기법을 활용해 새로운 파일로 저장
│   ├─ submission_viewer.ipynb # 제출용 submission csv 파일을 읽어 시각적으로 확인할 수 있게 해주는 viewer
│   └─ viewer.ipynb # 조건에 따라 이미지 단위, 오브젝트 단위로 이미지를 볼 수있는 viewer
│
├── work_dirs/ # 작업 결과가 저장되는 경로  
│   └─ *listup/ # 작업 결과들 중에 공유가 필요하다고 선택된 출력파일을 폴더 단위로 업로드
│
└── mmdetection/ # import mmdetection library in this path
```
- mmdetection의 load_from 이 절대경로밖에 지원하지 않아 object-detection-level2-cv-06이 detection이 되어야합니다. (`/opt/ml/detection/`)
- `work_dirs/listup/` 내의 각 폴더는 [\*.log, \*.py, \*.pth] 폴더를 포함한다.
  - \*.log : 최종 출력 로그
  - \*.py : 최종 config 파일
  - \*.pth : 최종 제출에 사용된 checkpoint

## Project Roadmap

![프레젠테이션1](https://user-images.githubusercontent.com/55044675/138575713-e6060134-a406-4cea-83f7-f5bd0b214058.png)

## best result
- /mmconfig/test/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco.py
  - Public : 0.620, private : 0.595
  - **universenet -** [weight download](https://drive.google.com/drive/folders/1pm8GUrOOILf8XKatn6MMISuEjdYr2xu5?usp=sharing)
- /mmconfig/listup/cascade_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_aug.py
  - Public : 0.650, private : 0.633
  - **swin_heavy_aug -** [weight download](https://drive.google.com/drive/folders/1MmG0_ExljbX1v6EKfUfKtCy-3ezs-e8N)
- ensemble universenet swin
  - iou_thr 0.6
  - Public : 0.687, private : 0.667

## simple start

### Train
`python ./mmdetection/tools/train.py /mmconfig/test/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco.py`

### Inference
`python ./utils/make_submission.py ./work_dirs/listup/{output path}/{config py path} {pth filename}`

### ensemble
`python ./utils/final_ensemble.py --iou_thr {float} ...`
- `utils/ensemble_models/` 경로에 앙상블 할 csv 파일들이 있어야함
