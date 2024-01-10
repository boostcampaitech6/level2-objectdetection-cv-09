import torch 
import numpy as np
from tqdm import tqdm
from dataset import TrainDataset, TestDataset, get_train_transform, get_valid_transform, get_test_transform, collate_fn
import models
from torch.utils.data import DataLoader

from infer import inference_fn, predict
import argparse
from omegaconf import OmegaConf
import os 
from copy import deepcopy

from metrics.mAP import mean_average_precision_for_boxes




def trainer(num_epochs, trainloader, validloader, optimizer, model, patience, checkpoint,device):
    # Train mode
    best_loss = float('inf')
    best_model = deepcopy(model.state_dict())
    # Patience count
    c = 0
    model.train()
    # Grad scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):

        total_train_loss = 0
        total_valid_loss = 0

        for images, targets, _ in tqdm(trainloader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 손실 계산
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                # calculate loss
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            total_train_loss += losses.item()

            # 역전파
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
      

        for images, targets, _ in tqdm(validloader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 손실 계산
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                # calculate loss
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            total_valid_loss += losses.item()
        
        train_loss = np.round(total_train_loss / len(trainloader),4)
        valid_loss = np.round(total_valid_loss / len(validloader),4)
        
        print(f'Epoch : {epoch + 1} Train loss : {train_loss}, Valid loss : {valid_loss}')

        # 모형 저장
        if valid_loss < best_loss:
            c = 0
            print('모형을 저장합니다')
            best_loss = valid_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, checkpoint)
        # 조기 종료
        else:
            c += 1
            if c == patience:
                print('학습 개선이 없으므로 학습을 조기 종료합니다.')
                break 






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/train.yaml"
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)

    # Dataset info
    train_annotation = configs['data']['train_annotation']
    valid_annotation = configs['data']['valid_annotation']
    data_dir = configs['data']['data_dir']
    
    # transforms
    train_transform = get_train_transform()
    valid_transform = get_valid_transform()

    # Dataset
    trainset = TrainDataset(train_annotation, data_dir, train_transform)
    validset = TrainDataset(valid_annotation, data_dir, valid_transform)

    batch_size = configs['batch_size']
    # Dataloader
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, collate_fn = collate_fn)
    validloader = DataLoader(validset, batch_size = batch_size, shuffle=False, collate_fn = collate_fn)

    device = configs['device']
    model_name = configs['train']['model']

    # 모형 불러오기
    if model_name == 'Faster_R-CNN_ResNet50_FPN':
        model = models.Faster_R_CNN_ResNet50_FPN(11)
        model.to(device)

    # Optimizer setting
    optim = configs['train']['optimizer']
    lr = configs['train']['learning_rate']
        
    params = [p for p in model.parameters() if p.requires_grad]

    if optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr)

    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr)

    # Epoch
    num_epochs = configs['train']['epoch']
    patience = configs['train']['patience']
    ckpt = configs['train']['checkpoint']

    # If there is no path checkpoint, make directory
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    print('학습 시작')
    print('-'*50)
    trainer(num_epochs, trainloader, validloader, optimizer, model, patience, ckpt,device)
    print('-'*50)
    print('학습 끝')

    # Test transform
    test_transforms = get_test_transform()

    # Validset
    validset = TestDataset(valid_annotation, data_dir,test_transforms)
    # Dataloader
    validloader =  DataLoader(validset, batch_size=16)
    # 최종 가중치 불러오기
    model.load_state_dict(torch.load(ckpt))
    # 평가
    model.eval()

    # 추론
    print('추론 및 예측')
    score_threshold = configs['eval']['score_threshold']
    outputs = inference_fn(validloader, model, device)
    # 예측
    prediction_strings, file_names = predict(valid_annotation, outputs, score_threshold)

    bboxes = prediction_strings.copy()

    # 예측 값 저장
    new_pred = []

    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for file_name, bbox in zip(file_names, bboxes):
        boxes = np.array(str(bbox).strip().split(' '))
        
        # boxes - class ID confidence score xmin ymin xmax ymax
        if len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        elif isinstance(bbox, float) or not bbox:
            #print(f'{file_name} empty box')
            continue
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
    
    # 정답 값 저장
    gt = []

    coco = validset.coco

    for image_id in coco.getImgIds():
        
        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)
            
        file_name = image_info['file_name']
            
        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

    # Calculate mAP
    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)