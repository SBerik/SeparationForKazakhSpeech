import torch
import lightning as L
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as sisnr

from .modules import Encoder, Decoder
from .sepformer_block import SepFormerBlock
from .utils import select_norm
from utils import tensify

# Testing 
# from modules import Encoder, Decoder
# from sepformer_block import SepFormerBlock
# from utils import select_norm


class DevSepfomer(torch.nn.Module):
    def __init__(self, 
                 out_channels = 256,   
                 encoder_type_norm='ln', 
                 num_layers_intra=8, 
                 num_layers_inter=8,
                 num_heads_intra=8, 
                 num_heads_inter=8,
                 d_intra=256, 
                 d_inter=256,
                 d_ff_intra=1024, 
                 d_ff_inter=1024, 
                 num_layers=2,
                 dropout = 0.0, 
                 K = 250,
                 speaker_num = 2,
                 optim_params = None, scheduler = None, clip_norm = None, training = None):
        
        super(DevSepfomer, self).__init__()
        self.K = K
        self.speaker_num = speaker_num
        self.norm_after_encoder = select_norm(encoder_type_norm, out_channels, 3)
        self.num_layers = num_layers
        self.sepformer = torch.nn.ModuleList([]) 
        
        
        for i in range(self.num_layers):
            self.sepformer.append(SepFormerBlock(num_layers_intra=num_layers_intra, 
                                                 num_layers_inter=num_layers_inter, 
                                                 num_heads_intra=num_heads_intra, 
                                                 num_heads_inter=num_heads_inter, 
                                                 d_intra=d_intra, 
                                                 d_inter=d_inter,
                                                 d_ff_intra=d_ff_intra, 
                                                 d_ff_inter=d_ff_inter,
                                                 dropout=dropout))
            # self.sepformer.append(DPTBlock(out_channels, H, self.Local_B, dropout))

        self.conv2d = torch.nn.Conv2d(out_channels, 
                                      out_channels*speaker_num, 
                                      kernel_size=1)
        self.prelu = torch.nn.PReLU()
        self.activation = torch.nn.ReLU()
        # gated output layer
        self.output = torch.nn.Sequential(torch.nn.Conv1d(out_channels, out_channels, 1),
                                          torch.nn.Tanh())
        self.output_gate = torch.nn.Sequential(torch.nn.Conv1d(out_channels, out_channels, 1),
                                               torch.nn.Sigmoid())

        self.sisnr_criterion = PIT(sisnr)
        self.sdr_criterion = PIT(sdr) 
        self.alpha, self.beta = (training['alpha'], training['beta']) if training is not None else (0.5, 0.5)
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.clip_norm = clip_norm

    def forward(self, x):
        '''
           x: [B, N, L]
        '''
        # [B, N, L]
        x = self.norm_after_encoder(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        x = x.permute(0, 1, 3, 2)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.sepformer[i](x)
        x = x.permute(0, 1, 3, 2)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B*self.speaker_num,-1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.speaker_num, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)
        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class PL_SuperiorSepformer(L.LightningModule):
    def __init__(self, 
                out_channels = 256,   
                kernel_size = 16,
                encoder_type_norm ='ln', 
                num_layers_intra = 8, 
                num_layers_inter = 8,
                num_heads_intra = 8, 
                num_heads_inter = 8,
                d_intra = 256, 
                d_inter = 256,
                d_ff_intra = 1024, 
                d_ff_inter = 1024, 
                num_layers = 2,
                dropout = 0.0, 
                K = 250,
                speaker_num = 2):
        

        super(PL_SuperiorSepformer, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size,
                               out_channels=out_channels, 
                               bias = False)
    
        self.separation = DevSepfomer(out_channels = out_channels,   
                                      encoder_type_norm=encoder_type_norm, 
                                      num_layers_intra=num_layers_intra, 
                                      num_layers_inter=num_layers_inter,
                                      num_heads_intra=num_heads_intra, 
                                      num_heads_inter=num_heads_inter,
                                      d_intra=d_intra, 
                                      d_inter=d_inter,
                                      d_ff_intra=d_ff_intra, 
                                      d_ff_inter=d_ff_inter, 
                                      num_layers=num_layers,
                                      dropout=dropout, 
                                      K=K,
                                      speaker_num=speaker_num)
        
        self.decoder = Decoder(in_channels=out_channels, 
                               out_channels=1, 
                               kernel_size=kernel_size, 
                               stride=kernel_size//2, 
                               bias=False)
        
        self.speaker_num = speaker_num
    
    def forward(self, x):
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        # [spks, B, N, L]
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        out = [s[i]*e for i in range(self.speaker_num)]
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

        # scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        # if self.trained_model and os.path.exists(self.trained_model):
        #     print(f"Loading pretrained model: {self.trained_model}", '\n')
        #     checkpoint = th.load(self.trained_model, map_location=self.device)
        #     optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return optimizer

if __name__ == "__main__":
    """
    model-params:
        in_channels | N: 256 
        speakers-num | C: 2  
        kernel-size | L: 8 
        heads-num | H: 8 
        chunks-size | K: 250
        Global_B | num_layers: 2 
        Local_B: 8
    """

    model = PL_SuperiorSepformer(out_channels=256,   
                              kernel_size=16,
                              encoder_type_norm='ln', 
                              num_layers_intra=8, 
                              num_layers_inter=8,
                              num_heads_intra=8, 
                              num_heads_inter=8,
                              d_intra=256, 
                              d_inter=256,
                              d_ff_intra=1024, 
                              d_ff_inter=1024, 
                              num_layers=2,
                              dropout=0.0, 
                              K=250,
                              speaker_num=2)
    
    x = torch.ones(1, 16001)
    out = model(x)
    
    print(out[0].shape)