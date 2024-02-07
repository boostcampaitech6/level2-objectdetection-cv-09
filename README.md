# level- 2 대회 :재활용 품목 분류를 위한 Object Detection 대회

## 팀 소개
| 이름 | 역할 |
| ---- | --- |
| [박상언](https://github.com/PSangEon) | Detectron2 baseline code 작성, 모자이크 코드 작성, 트랜스포머 모형 실험 |
| [지현동](https://github.com/tolfromj) | EDA, MMDectection 실험, 증강 기법 실험, Anchor box 실험, 모형 실험, Git 현황 관리 |
| [오왕택](https://github.com/ohkingtaek) | EDA, YOLO, MMDetection baseline code 작성, 시각화 코드 작성, 증강 기법 실험,</br>수도 라벨링, 모형 실험, 앙상블 실험 |
| [이동호](https://github.com/as9786) | Torchvision baseline code 작성, Detectron2 실험, Anchor box 실험, WBF code 작성, 모형 실험 |
| [송지민](https://github.com/Remiing) | EDA, YOLO, MMDetection baseline code 작성, 시각화 코드 작성 |
| [이주헌](https://github.com/LeeJuheonT6138) | Detectron2 실험, YOLO 실험, Git 현황 관리 |

## 프로젝트 소개
<p align="center">
<img src="https://github.com/boostcampaitech6/level2-objectdetection-cv-09/assets/49676680/6f27ddee-6f75-4a61-b3d7-31c2951a0235">
</p>

우리는 사진에서 쓰레기를 탐지하는 모형을 구축하여 정확한 분리수거를 도울 수 있는 프로젝트를 진행하였다. 데이터셋은 사진과 annotation file로 구성이 되어있다. Annotation file에는 사진의 크기, 클래스, 경계 상자의 좌표, 경계 상자의 너비와 높이 등사진의 정보들이 담겨 있다. 클래스는 쓰레기, 플라스틱, 종이, 유리 등 총 10개의 클래스를 분류하고, 해당 클래스가 존재하는 좌표 값을 구해야 했다.  

## 프로젝트 일정
프로젝트 전체 일정
- 01/13 10:00 ~ 01/18 19:00

프로젝트 세부 일정
- 01/03 ~ 01/05 강의 수강, 제공 데이터 및 코드 확인
- 01/06 ~ 01/10 BaseLine Code 작성, Git Branch 생성, EDA
- 01/11 ~ 01/14 MMDetection, YOLO로 라이브러리 결정, 시각화 코드 작성, 모형 실험, 증강 기법 실험, Validation set setting
- 01/15 ~ 01/18 Hyperparameter tuning, Anchor box 조정, 모형 실험, WBF, Pseudo-labeling

## 프로젝트 수행
- EDA : 클래스 불균형 확인, 경계박스 비율 확인
- Baseline code, 사용할 라이브러리 선택 : MMDetection, Detectron, YOLO, torchvision 모두 실험
- Test Set과 비슷한 지표를 가지는 Validation Set 찾기 : StratifiedKFold 사용
- 여러 가지 증강 기법 실험 : Clahe, Mosaic, Albumentation등 다양한 기법 적용
- 다양한 모형 실험 : Faster R-CNN, Cascade R-CNN, DINO, Dynamic Head 실험
- Anchor Box Adjusting : EDA한 결과로 Anchor Box 조정해 비교 실험
- Weighted Boxes Fusion : 앙상블 기법으로 WBF 사용
- Pseudo Labeling : 테스트 데이터를 추론하고 학습해 성능 향상

## 프로젝트 결과
- 프로젝트 결과는 Public 4등, Private 3등이라는 결과를 얻었습니다.
![](https://github.com/boostcampaitech6/level2-objectdetection-cv-09/assets/49676680/bd476ae8-2d62-4cb0-bb49-99c92b22176c)
![](https://github.com/boostcampaitech6/level2-objectdetection-cv-09/assets/49676680/328b1ab6-d9f4-41f3-aeb3-e9fe1e403fc6)


## Wrap-Up Report

- [Wrap-Up Report](https://github.com/boostcampaitech6/level2-objectdetection-cv-09/blob/354361d4feaaadf95b188c8d9957090641bbe2db/docs/Object%20Det_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(09%EC%A1%B0).pdf)

## File Tree

```bash
.
├── configs
│   └── _custom_
├── docs
│   └── Object Det_CV_팀 리포트(09조).pdf
├── notebooks
│   ├── data_split.ipynb
│   ├── offline_nms.ipynb
│   ├── pseudo_labeling.ipynb
│   └── WBF.ipynb
├── tools
│   ├── train.py
│   └── test.py
├── work_dirs
├── train.sh
├── eval.sh
└── WBF.py
```

| File(.py) | Description |
| --- | --- |
| data_split.ipynb | StratifiedKFold로 train/valid 분할 |
| offline_nms.ipynb | 데이터에 오프라인으로 NMS 적용하기 |
| pseudo_labeling.ipynb | pseudo labeling하기 |
| WBF.ipynb | Weighted Boxes Fusion 코드 |
| WBF.py | Weighted Boxes Fusion 코드 |
| train.sh | train sh파일 |
| test.sh | test sh파일 |
| train.py | train 코드 |
| test.py | test 코드 |

## License
네이버 부스트캠프 AI Tech 교육용 데이터로 대회용 데이터임을 알려드립니다.
