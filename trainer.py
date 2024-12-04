import torch
from utils.measure_time import measure_time
from utils.training import * 
import os

class Trainer:
    def __init__(self, num_epochs = 100, device='cuda', best_weights = False, checkpointing = False, 
                 checkpoint_interval = 10, model_name = '', path_to_weights= './weights', ckpt_folder = '') -> None:
        self.num_epochs = num_epochs
        self.device = device
        self.best_weights = best_weights
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        os.makedirs(path_to_weights, exist_ok=True)
        self.path_to_weights = path_to_weights
        self.ckpt_folder = ckpt_folder

    @measure_time
    def fit(self, model, dataloaders, criterion, optimizer, metrics, writer) -> None:
        model.to(self.device)
        min_acc = 0.0
        for epoch in range(self.num_epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                    dataloader = dataloaders['train']
                else:
                    model.eval()
                    dataloader = dataloaders['valid']
                
                running_loss = 0.0
                for m in metrics.keys():
                    metrics[m].reset()
                total_samples = len(dataloader.dataset)
                
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).transpose(1, 2)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    for m in metrics.keys():
                        metrics[m].update(outputs, labels)
                
                epoch_loss = running_loss / total_samples
                epoch_metrics = {m: metrics[m].compute().item() for m in metrics.keys()}
                
                torch_logger (writer, phase, epoch, epoch_loss, epoch_metrics, metrics)
                p_output_log(epoch, self.num_epochs, phase, epoch_loss, epoch_metrics, metrics)

                if phase == 'valid' and self.best_weights and epoch_metrics['Accuracy'] > min_acc:
                    min_acc = epoch_metrics['Accuracy']
                    save_best_weight(model, optimizer, epoch, epoch_loss, epoch_metrics, self.path_to_weights, self.model_name)

            if self.checkpointing and (epoch + 1) % self.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_metrics, self.ckpt_folder, self.model_name)
