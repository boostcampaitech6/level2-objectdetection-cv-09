import torch 
import numpy as np
from tqdm import tqdm
from dataset import TrainDataset, get_train_transform, collate_fn
import models
from torch.utils.data import DataLoader

import argparse
from omegaconf import OmegaConf
import os 


def trainer(num_epochs, trainloader, optimizer, model, device):
    # Train mode
    model.train()
    best_loss = float('inf')
    
    # Grad scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):

        total_loss = 0

        for images, targets, _ in tqdm(trainloader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 손실 계산
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                # calculate loss
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()

            # 역전파
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        
        print(f'Epoch : {epoch + 1} loss : {np.round(total_loss / len(trainloader),4)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/train.yaml"
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)

    
    annotation = configs['data']['annotation']
    data_dir = configs['data']['data_dir']
    transform = get_train_transform()

    trainset = TrainDataset(annotation, data_dir, transform)

    batch_size = configs['batch_size']
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, collate_fn = collate_fn)

    device = configs['device']
    model_name = configs['train']['model']

    if model_name == 'Faster_R-CNN_ResNet50_FPN':
        model = models.Faster_R_CNN_ResNet50_FPN(11)
        model.to(device)

    optim = configs['train']['optimizer']
    lr = configs['train']['learning_rate']
        
    params = [p for p in model.parameters() if p.requires_grad]

    if optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr)

    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr)

    num_epochs = configs['train']['epoch']

    print('Training starts\n')
    trainer(num_epochs, trainloader, optimizer, model, device)
    
    print('\nTraining finish')

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
        
    ckpt = configs['train']['checkpoint']
    torch.save(model.state_dict(), ckpt)
