import torch
import torchvision
from torch.utils.data import Dataset 

from pycocotools.coco import COCO

import numpy as np
import cv2
import os

import albumentations as A 
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ],bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    return tuple(zip(*batch))

    
class TrainDataset(Dataset):
    '''
    Make training dataset
    annotation : train.json path
    data_dir : dataset path
    '''
    def __init__(self, annotation, data_dir, transforms=None):

        super().__init__()
        self.data_dir = data_dir
        # Load COCO annotation
        self.coco = COCO(annotation)
        self.transforms = transforms 


    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx:int):

        # Get ID of image
        ids = self.coco.getImgIds()
        image_id = self.coco.getImgIds(imgIds=ids[idx])
        # Get info about image ex) Width, height, id etc
        image_info = self.coco.loadImgs(image_id)[0]

        # Open image
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        # Convert to RGB shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 정규화
        image /= 255.0

        # 정답 정보 
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        # Get annotation info ex) bbox, categories etc
        anns = self.coco.loadAnns(ann_ids)

        # 정답 경계 상자 좌표 얻기
        boxes = np.array([x['bbox'] for x in anns])

        # Adjust bbox offset(x_min, y_min, x_max, y_max)
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]

        # Torchvision faster R-CNN은 label = 0이 배경
        # class_id를 1~10 값으로 변경
        labels = np.array([x['category_id']+1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # 겹쳐 있는 정보
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes' : boxes, 'labels' : labels, 'image_id' : torch.tensor([idx]), 
                'area' : areas, 'iscrowd' : is_crowds}

        # Transform
        if self.transforms:
            sample = {'image':image, 'bboxes' : target['boxes'], 'labels' : labels}
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'],dtype=torch.float32)

        return image, target, image_id



class TestDataset(Dataset):
    ''' 
    Make test dataset
    annotation : test.json
    data_dir : dataset path
    '''

    def __init__(self, annotation, data_dir, transforms):
        super().__init__()
        self.data_dir = data_dir 
        self.coco = COCO(annotation)
        self.transforms = transforms

    def __len__(self) -> int: 
        return len(self.coco.getImgIds())

    def __getitem__(self, idx: int):
        # Get ID of image
        ids = self.coco.getImgIds()
        image_id = self.coco.getImgIds(imgIds=ids[idx])
        # Get image info
        image_info = self.coco.loadImgs(image_id)[0]
        # Open image
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        # Convert RGB color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 정규화
        image /= 255.0

        # 
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

        return image