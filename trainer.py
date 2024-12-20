import os

import torch

from utils.measure_time import measure_time
from utils.checkpointer import Checkpointer
from utils.training import * 

class Trainer:
    def __init__(self, num_epochs = 100, device='cuda', best_weights = False, checkpointing = False, 
                 checkpoint_interval = 10, model_name = '', trained_model = './', path_to_weights= './weights', 
                 ckpt_folder = '', speaker_num = 2, resume = False) -> None:
        self.num_epochs = num_epochs
        self.device = device
        self.best_weights = best_weights
        self.ckpointer = Checkpointer(model_name, path_to_weights, ckpt_folder, metrics = False)
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        os.makedirs(path_to_weights, exist_ok=True)
        self.path_to_weights = path_to_weights
        self.ckpt_folder = ckpt_folder
        self.speaker_num = speaker_num
        self.resume = resume
        self.trained_model = trained_model

    @measure_time
    def fit(self, model, dataloaders, criterion, optimizer, writer) -> None:
        model.to(self.device)
        start_epoch, min_val_loss, model = self.load_pretrained_model(model, optimizer)
        epoch_state = EpochState(metrics = None)
        for epoch in range(start_epoch, self.num_epochs):
            for phase in ['train', 'valid']:
                model.train() if phase == 'train' else model.eval()
                dataloader = dataloaders[phase] 
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
                epoch_state.update_state(epoch_loss, phase)
                p_output_log(self.num_epochs, epoch, phase, epoch_state)
                
                if phase == 'valid' and self.best_weights and epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    self.ckpointer.save_best_weight(model, optimizer, epoch, epoch_state)
            
            torch_logger(writer, epoch, epoch_state)
            
            if self.checkpointing and (epoch + 1) % self.checkpoint_interval == 0:
                self.ckpointer.save_checkpoint(model, optimizer, epoch, epoch_state)

    def load_pretrained_model(self, model, optimizer):
        if self.trained_model:
            print(f"Load pretrained mode: {self.trained_model}", '\n')
            checkpoint = torch.load(self.trained_model, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1, checkpoint['val_loss'] , model
        else:
            return 0, float('inf'), model