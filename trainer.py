import torch
from utils.measure_time import measure_time
from utils.training import * 
import os
import time

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
        for epoch in range(self.num_epochs):
            print('Epoch:', epoch)
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                    dataloader = dataloaders['train']
                else:
                    model.eval()
                    dataloader = dataloaders['valid']
                
                running_loss = 0.0
                total_samples = len(dataloader.dataset)
                
                for mixed, ref in dataloader:
                    inputs = mixed.to(self.device)
                    labels = [ref[i].to(self.device) for i in range(self.num_spks)]
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    
                epoch_loss = running_loss / total_samples
                
                if phase == 'valid':
                    print('valid:', epoch_loss)
                else:
                    print('train', epoch_loss)