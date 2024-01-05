# Torchvision model for object detection

## 학습

### config/train.yaml
- annotation : train.json 경로
- data_dir : Data들이 있는 경로
- device : 학습 장치 설정
- batch_size : 배치 크기
- epoch : 학습할 횟수
- optimizer : 'Adam', 'SGD', 'AdamW' 중 하나 입력하여 함수 설정
- learning_rate : 학습율
- model : 사용할 모형(현재는 'Faster_R-CNN_ResNet50_FPN' 하나만 있음)
- checkpoint : 가중치를 저장할 경로

### Augmentation

- dataset.py에 가서 get_train_transform() 함수에서 인자 조정

### 학습
'''{python}
python train.py
'''


## 예측

### config/infer.yaml
- annotation : test.json 경로
- data_dir : 데이터셋 경로
- batch_size : 배치 크기
- checkpoint : 학습 시 저장했던 가중치 경로
- model : 예측에 사용할 모형 선택
- device : 장치 설정
- score_threshold : 신뢰 점수 하한 값
- save_dir : 제출 파일 저장 경로

### 예측
```{python}
python infer.py
```
