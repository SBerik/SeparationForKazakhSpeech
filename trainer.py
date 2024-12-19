import os

import torch
from utils.measure_time import measure_time
from utils.training import * 

class Trainer:
    def __init__(self, num_epochs = 100, device='cuda', best_weights = False, checkpointing = False, 
                 checkpoint_interval = 10, model_name = '', path_to_weights= './weights', ckpt_folder = '',
                 speaker_num = 2, resume = False) -> None:
        self.num_epochs = num_epochs
        self.device = device
        self.best_weights = best_weights
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        os.makedirs(path_to_weights, exist_ok=True)
        self.path_to_weights = path_to_weights
        self.ckpt_folder = ckpt_folder
        self.speaker_num = speaker_num
        self.resume = resume

    @measure_time
    def fit(self, model, dataloaders, criterion, optimizer, writer) -> None:
        model.to(self.device)
        min_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            for phase in ['train', 'valid']:
                model.train() if phase == 'train' else model.eval()
                dataloader = dataloaders[phase] 
                print(phase, 'dataloader len', len(dataloader))
                running_loss = 0.0
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = [l.to(self.device) for l in labels]
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                epoch_loss = running_loss / len(dataloader.dataset)
                print('epoch_loss', epoch_loss)
                # break
                # p_output_log(self.num_epochs, epoch, epoch_loss)