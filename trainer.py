import torch
from utils.measure_time import measure_time
from utils.training import * 
import os

class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, Dual_RNN,  optimizer, scheduler, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.name = opt['name']
        if opt['train']['gpuid']:
            self.gpuid = opt['train']['gpuid']
            self.device = torch.device(
                'cuda:{}'.format(opt['train']['gpuid'][0]))
            self.dualrnn = Dual_RNN.to(self.device)
        self.dualrnn = Dual_RNN.to(self.device)
        self.optimizer = optimizer
        self.clip_norm = 0

    def train(self, epoch):
        self.dualrnn.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        for mix, ref in self.train_dataloader:
            mix = mix.to(self.device)
            ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            self.optimizer.zero_grad()

            if self.gpuid:
                out = torch.nn.parallel.data_parallel(self.dualrnn,mix,device_ids=self.gpuid)
                #out = self.dualrnn(mix)
            else:
                out = self.dualrnn(mix)

            l = Loss(out, ref)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.dualrnn.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss


class MyTrainer:
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
