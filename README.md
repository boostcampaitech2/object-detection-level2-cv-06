# 칠성사이다

## Requirements
- python, numpy, pandas
- pytorch
- mmdetection
- torchvision
- albumentations
- seaborn

## contents
```
object-detection-level2-cv-06/
├── EDA/ # EDA files
├── dataset/
│   ├─ candidate/ # valid candidate
│   ├─ sample.json # just sample
│   ├─ test.json # orginal test.json, full data
│   └─ train.json # orginal train.json, full data
├── mmconfig/ # mmdetection용 config 모음
│   ├─ datasets/ # dataset configs
│   ├─ models/ # model configs
│   │  └─ pretrained/ # pretrained save pth path
│   ├─ schedules/ # schedule configs
│   ├─ runtime/ # runtime configs
│   ├─ sample/ # old style configs, current not used
│   ├─ listup/ # test config에서 선택되어 분류된 config들
│   └─ test/ # [datasets, models, schedules, runtime] config 들을 통합한 main config
├── utils/ # 부가적으로 필요한 util들  
│   ├─ EDA.ipynb # main EDA
│   ├─ anno_converter.ipynb # coco dataset json 파일을 조건에따라 샘플링하여 파일로 저장해주는 converter
│   └─ viewer.ipynb # 조건에 따라 이미지 단위, 오브젝트 단위로 이미지를 볼 수있는 viewer
├── work_dirs/ # 작업 결과가 저장되는 경로  
│   └─ *listup/ # 작업 결과들 중에 공유가 필요하다고 선택된 출력파일을 폴더 단위로 업로드
└── mmdetection/ # import mmdetection library in this path
```
- `work_dirs/listup/` 내의 각 폴더는 [\*.log, \*.py, \*.pth] 폴더를 포함한다.
  - \*.log : 최종 출력 로그
  - \*.py : 최종 config 파일
  - \*.pth : 최종 제출에 사용된 checkpoint
