import torch
import pytorch_lightning as pl
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as sisnr

from .modules import Encoder, Decoder
from .dualpathrnn import Dual_Path_RNN
from utils.training import tensify


class PL_Dual_RNN_model(pl.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, speaker_num=2000,
                 optim_params = None, scheduler = None, clip_norm = None, training = None):  # To do - исправить None
        super(PL_Dual_RNN_model, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=in_channels, bias=False)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                                        rnn_type=rnn_type, norm=norm, dropout=dropout,
                                        bidirectional=bidirectional, num_layers=num_layers, 
                                        K=K, speaker_num=speaker_num)
        self.decoder = Decoder(in_channels=in_channels, out_channels=1, 
                               kernel_size=kernel_size, stride=kernel_size // 2, bias=False)
        self.speaker_num = speaker_num
        self.sisnr_criterion = PIT(sisnr)
        self.sdr_criterion = PIT(sdr) 
        self.alpha, self.beta = (training['alpha'], training['beta']) if training is not None else (0.5, 0.5)
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.clip_norm = clip_norm

    def forward(self, x):
        # x: [B, L]
        e = self.encoder(x) # output # [B, N, L]
        s = self.separation(e) # output [spks, B, N, L]
        out = [s[i]*e for i in range(self.speaker_num)] # [B, N, L] -> [B, L]
        audio = [self.decoder(out[i]) for i in range(self.speaker_num)]
        return audio

    def epoch_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        labels, outputs = tensify(labels), tensify(outputs) if not isinstance(outputs, torch.Tensor) else outputs
        sisnr_loss = -self.sisnr_criterion(outputs, labels)
        sdr_loss = -self.sdr_criterion(outputs, labels)
        loss = self.alpha * sisnr_loss + self.beta * sdr_loss
        return {'sisnr': sisnr_loss, 'sdr': sdr_loss, 'loss': loss}

    def log_losses(self, losses, mode='train'):
        for loss_name, loss_value in losses.items():
            if loss_name == 'loss':
                self.log(f"{mode}_loss", loss_value, on_epoch=True, prog_bar=True, logger=True)
            else:
                self.log(f'{mode}_{loss_name}', loss_value, on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx):
        losses = self.epoch_step(batch)
        self.log_losses(losses, mode="train")
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        losses = self.epoch_step(batch)
        self.log_losses(losses, mode="val")
        return losses['loss'] 

    def configure_optimizers(self):
        optim_type = self.optim_params["type"]
        assert optim_type in ['Adam', 'SGD'], "Invalid optimizer type"
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_params[optim_type]['lr'])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        # if self.trained_model and os.path.exists(self.trained_model):
        #     print(f"Loading pretrained model: {self.trained_model}", '\n')
        #     checkpoint = torch.load(self.trained_model, map_location=self.device)
        #     optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return optimizer

        # This method is needed to track loss for ReduceLROnPlateau or ClipNorm
        # return {
        #     "optimizer": optimizer,
        #     "monitor": "val_loss"  # Параметр monitor теперь на том же уровне
        # }
    

    # To Do
    # For testing on test dataset ;xd
    # def test_step(self, batch, batch_idx):
        # pass 
    # def on_test_start(self):
    #     pass
    # def on_test_end(self):
    #     pass

    # For inference 
    # def predict_step(batch, batch_idx):
        # pass


if __name__ == '__main__':
    rnn = PL_Dual_RNN_model(256, 64, 128, bidirectional=True, norm='ln', num_layers=6)
    x = torch.ones(1, 100)
    out = rnn.forward(x)