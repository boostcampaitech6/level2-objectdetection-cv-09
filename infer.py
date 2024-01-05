from dataset import TestDataset, get_valid_transform
import models
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 

from pycocotools.coco import COCO

import argparse
from omegaconf import OmegaConf



def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def predict(annotation, outputs, score_threshold):
    
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    return prediction_strings, file_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/infer.yaml"
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)

    annotation = configs['data']['annotation']
    data_dir = configs['data']['data_dir']
    transforms = get_valid_transform()
    batch_size = configs['batch_size']

    testset = TestDataset(annotation, data_dir, transforms)
    testloader = DataLoader(testset, batch_size=batch_size,shuffle=False)

    device = configs['device']

    model = models.Faster_R_CNN_ResNet50_FPN(11)
    model.to(device)

    checkpoint = configs['checkpoint']
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    outputs = inference_fn(testloader, model, device)
    score_threshold = configs['score_threshold']

    prediction_strings, file_names = predict(annotation, outputs, score_threshold)


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    save_dir = configs['save_dir']
    submission.to_csv(save_dir, index=False)

