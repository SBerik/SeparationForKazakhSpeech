import torch
import torch.nn.functional as F
import math
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module

from .modules import Encoder, Decoder
# from modules import Encoder, Decoder

from .sepformer_block import SepFormerBlock
# from sepformer_block import SepFormerBlock


# ============================ Normalization code-block

class GlobalLayerNorm(torch.nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = torch.nn.Parameter(torch.ones(self.dim, 1))
                self.bias = torch.nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = torch.nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = torch.nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(torch.nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return torch.nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return torch.nn.BatchNorm1d(dim)

# ============================ Normalization end block


class DevSepfomer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  norm='ln', dropout=0.1, 
                K=200, speaker_num=2, H = 8, num_layers = 2, Local_B = 8):
        super(DevSepfomer, self).__init__()
        d_model = out_channels
        d_ff = 1024
        self.K = K
        self.speaker_num = speaker_num
        self.norm = select_norm(norm, in_channels, 3)
        self.num_layers = num_layers
        self.Local_B = Local_B
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.sepformer = torch.nn.ModuleList([]) # changed
        for i in range(self.num_layers):
            self.sepformer.append(SepFormerBlock(d_intra=d_model, d_inter=d_model, d_ff_intra=d_ff, d_ff_inter=d_ff))
            # self.sepformer.append(DPTBlock(out_channels, H, self.Local_B, dropout))

        self.conv2d = torch.nn.Conv2d(
            out_channels, out_channels*speaker_num, kernel_size=1)
        self.end_conv1x1 = torch.nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = torch.nn.PReLU()
        self.activation = torch.nn.ReLU()
         # gated output layer
        self.output = torch.nn.Sequential(torch.nn.Conv1d(out_channels, out_channels, 1),
                                    torch.nn.Tanh()
                                    )
        self.output_gate = torch.nn.Sequential(torch.nn.Conv1d(out_channels, out_channels, 1),
                                         torch.nn.Sigmoid()
                                         )

    def forward(self, x):
        '''
           x: [B, N, L]

        '''
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        # x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # print('beform permute()', x.shape, gap)
        x = x.permute(0, 1, 3, 2)
        # print('after permute()', x.shape, gap)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.sepformer[i](x)
        x = x.permute(0, 1, 3, 2)
        # print('end again', x.shape, gap)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B*self.speaker_num,-1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
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


class SuperiorSepformer(torch.nn.Module):
    def __init__(self, in_channels = 256, out_channels = 64, kernel_size=2,  norm='ln', dropout=0,
                 K=200, speaker_num=2, H = 8, num_layers = 2, Local_B = 8):
        super(SuperiorSepformer, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size,
                               out_channels=in_channels, 
                               bias = False)
        
        self.separation = DevSepfomer(in_channels, 
                                      out_channels,  
                                      norm=norm, 
                                      dropout=dropout,  
                                      K=K, 
                                      speaker_num=speaker_num,
                                      H = H, 
                                      num_layers=num_layers,
                                      Local_B = Local_B)
        
        self.decoder = Decoder(in_channels=in_channels, 
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

    model = SuperiorSepformer(in_channels = 256, 
                      out_channels = 256,   
                      kernel_size=2,
                      norm='ln', 
                      dropout = 0, 
                      K = 250,
                      speaker_num = 2, 
                      H = 8,
                      num_layers =2,
                      Local_B= 8)
    
    #encoder = Encoder(16, 512)
    
    x = torch.ones(1, 16001)
    
    out = model(x)
    print(out[0].shape)