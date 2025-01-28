import torch
import torch.nn.functional as F
import math
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module

from .modules import Encoder, Decoder
# from modules import Encoder, Decoder


# ============================ Normalization blocks
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


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dropout=0):
        super(TransformerEncoderLayer, self).__init__()

        self.LayerNorm1 =torch.nn.LayerNorm(normalized_shape=d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.Dropout1 =torch.nn.Dropout(p=dropout)

        self.LayerNorm2 =torch.nn.LayerNorm(normalized_shape=d_model)

        self.FeedForward =torch.nn.Sequential(torch.nn.Linear(d_model, d_model*2*2),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(d_model*2*2, d_model))

        self.Dropout2 =torch.nn.Dropout(p=dropout)

    def forward(self, z):

        z1 = self.LayerNorm1(z)

        z2 = self.self_attn(z1, z1, z1, attn_mask=None, key_padding_mask=None)[0]

        z3 = self.Dropout1(z2) + z

        z4 = self.LayerNorm2(z3)

        z5 = self.Dropout2(self.FeedForward(z4)) + z3

        return z5


class Positional_Encoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(Positional_Encoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)  # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0)  # batch, channels, seq_len

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x.permute(0, 2, 1).contiguous()

        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]

        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]

        x = self.dropout(x)

        x = x.permute(0, 2, 1).contiguous()

        return x


class DPTBlock(torch.nn.Module):

    def __init__(self, input_size, nHead, Local_B, dropout = 0.1):

        super(DPTBlock, self).__init__()
        self.Local_B = Local_B
        self.intra_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.intra_transformer = torch.nn.ModuleList([])
        for i in range(self.Local_B):
            self.intra_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=dropout))

        self.inter_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.inter_transformer = torch.nn.ModuleList([])
        for i in range(self.Local_B):
            self.inter_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=dropout))

    def forward(self, z):

        B, N, K, P = z.shape

        # intra DPT
        row_z = z.permute(0, 3, 2, 1).contiguous().view(B*P, K, N)
        row_z1 = self.intra_PositionalEncoding(row_z)

        for i in range(self.Local_B):
            row_z1 = self.intra_transformer[i](row_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        row_f = row_z1 + row_z
        row_output = row_f.view(B, P, K, N).permute(0, 3, 2, 1).contiguous()

        # inter DPT
        col_z = row_output.permute(0, 2, 3, 1).contiguous().view(B*K, P, N)
        col_z1 = self.inter_PositionalEncoding(col_z)

        for i in range(self.Local_B):
            col_z1 = self.inter_transformer[i](col_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        col_f = col_z1 + col_z
        col_output = col_f.view(B, K, P, N).permute(0, 3, 1, 2).contiguous()

        return col_output



class DevSepfomer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  norm='ln', dropout=0.1, 
                K=200, speaker_num=2, H = 8, num_layers = 2, Local_B = 8):
        super(DevSepfomer, self).__init__()
        self.K = K
        self.speaker_num = speaker_num
        self.norm = select_norm(norm, in_channels, 3)
        self.num_layers = num_layers
        self.Local_B = Local_B
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.sepformer = torch.nn.ModuleList([]) # changed
        for i in range(self.num_layers):
            self.sepformer.append(DPTBlock(out_channels, H, self.Local_B, dropout))

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
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.sepformer[i](x)
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


class MySepfomer(torch.nn.Module):
    def __init__(self, in_channels = 256, out_channels = 64, kernel_size=2,  norm='ln', dropout=0,
                 K=200, speaker_num=2, H = 8, num_layers = 2, Local_B = 8):
        super(MySepfomer, self).__init__()
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

    model = MySepfomer(in_channels = 256, 
                      out_channels = 64,   
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