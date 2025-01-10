import os

import torch

from utils.measure_time import measure_time
from utils.checkpointer import Checkpointer
from utils.training import * 


class Trainer:
    def __init__(self, epochs = 100, device='cuda', best_weights = False, checkpointing = False, 
                 checkpoint_interval = 10, model_name = '', trained_model = './', path_to_weights= './weights', 
                 ckpt_folder = '', speaker_num = 2, resume = False, alpha = 0.5, beta = 0.5) -> None:
        self.epochs = epochs
        self.device = device
        self.best_weights = best_weights
        self.ckpointer = Checkpointer(model_name, path_to_weights, ckpt_folder)
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        os.makedirs(path_to_weights, exist_ok=True)
        self.path_to_weights = path_to_weights
        self.ckpt_folder = ckpt_folder
        self.speaker_num = speaker_num
        self.trained_model = trained_model
        self.alpha = alpha
        self.beta = beta

    @measure_time
    def fit(self, model, dataloaders, criterions, optimizer, writer) -> None:
        model.to(self.device)
        start_epoch, min_val_loss, model, optimizer = self.load_pretrained_model(model, optimizer)
        epoch_state = EpochState(metrics = criterions, epochs=self.epochs)
        for epoch in range(start_epoch, self.epochs):
            for phase in ['train', 'valid']:
                model.train() if phase == 'train' else model.eval()
                dataloader = dataloaders[phase] 
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), [l.to(self.device) for l in labels]
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        labels = tensify(labels).to(self.device) 
                        outputs = tensify(outputs).to(self.device) if not isinstance(outputs, torch.Tensor) else outputs.to(self.device)
                        losses = {'sisnr': - criterions['sisnr'](outputs, labels),
                                  'sdr': - criterions['sdr'](outputs, labels)}  
                        loss = self.alpha * losses['sisnr'] + self.beta * losses['sdr']
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    epoch_state.update_loss(phase, loss)
                    epoch_state.update_metrics(phase, losses)
                epoch_loss = epoch_state.compute_loss(phase, len(dataloader))
                epoch_metrics = epoch_state.compute_metrics(phase, len(dataloader))
                epoch_state.p_output(epoch, phase)
                if phase == 'valid' and self.best_weights and epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    self.ckpointer.save_best_weight(model, optimizer, epoch, epoch_state)
            torch_logger(writer, epoch, epoch_state)
            if self.checkpointing and (epoch + 1) % self.checkpoint_interval == 0:
                self.ckpointer.save_checkpoint(model, optimizer, epoch, epoch_state)
            epoch_state.reset_state()

    def load_pretrained_model(self, model, optimizer):
        if self.trained_model:
            print(f"Load pretrained mode: {self.trained_model}", '\n')
            checkpoint = torch.load(self.trained_model, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1, checkpoint['val_loss'] , model, optimizer
        else:
            return 0, float('inf'), model, optimizer