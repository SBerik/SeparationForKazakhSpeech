import torch.nn as nn
import torch


EPS = 1e-12

class SepFormerBlock(nn.Module):
    def __init__(
        self,
        num_layers_intra=8, num_layers_inter=8,
        num_heads_intra=8, num_heads_inter=8,
        d_intra=256, d_inter=256, d_ff_intra=1024, d_ff_inter=1024,
        norm=True, dropout=0.0, nonlinear='relu',
        causal=False,
        eps=EPS
    ):
        super().__init__()

        self.intra_transformer = IntraTransformer(
            d_intra,
            num_layers=num_layers_intra, num_heads=num_heads_intra, d_ff=d_ff_intra,
            norm=norm, dropout=dropout, nonlinear=nonlinear,
            eps=eps
        )
        self.inter_transformer = InterTransformer(
            d_inter,
            num_layers=num_layers_inter, num_heads=num_heads_inter, d_ff=d_ff_inter,
            norm=norm, dropout=dropout, nonlinear=nonlinear, causal=causal,
            eps=eps
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_transformer(input)
        output = self.inter_transformer(x)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout=0, max_len=5000, base=10000, batch_first=False):
        super().__init__()

        self.batch_first = batch_first

        position = torch.arange(max_len) # (max_len,)
        index = torch.arange(0, num_features, 2) / num_features # (num_features // 2,)        
        indices = position.unsqueeze(dim=1) / (base ** index.unsqueeze(dim=0)) # (max_len, num_features // 2)
        sin, cos = torch.sin(indices), torch.cos(indices)
        positional_encoding = torch.stack([sin, cos], dim=-1) # (max_len, num_features // 2, 2)

        if batch_first:
            positional_encoding = positional_encoding.view(max_len, num_features)
        else:
            positional_encoding = positional_encoding.view(max_len, 1, num_features)

        self.register_buffer("positional_encoding", positional_encoding)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        """
        Args:
            input: (T, batch_size, num_features) if batch_first=False, otherwise (batch_size, T, num_features)
        Returns:
            output: (T, batch_size, num_features) if batch_first=False, otherwise (batch_size, T, num_features)
        """
        if self.batch_first:
            T = input.size(1)
            x = input + self.positional_encoding[:, :T]
        else:
            T = input.size(0)
            x = input + self.positional_encoding[:T]

        output = self.dropout(x)

        return output


class IntraTransformer(nn.Module):
    def __init__(self, num_features, num_layers=8, num_heads=8, d_ff=1024, norm=True, nonlinear='relu', dropout=1e-1, norm_first=False, eps=EPS):
        super().__init__()

        self.num_features = num_features

        if isinstance(norm, int):
            if norm:
                norm_name = 'gLN'
                layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)
            else:
                layer_norm = None
        else:
            norm_name = norm
            layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)

        self.positional_encoding = PositionalEncoding(num_features, batch_first=False)
        encoder_layer = nn.TransformerEncoderLayer(num_features, num_heads, d_ff, dropout=dropout, activation=nonlinear, layer_norm_eps=eps, batch_first=False, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        # print('input.shape', input.shape)
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        residual = input
        x = input.permute(3, 0, 2, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (chunk_size, batch_size, S, num_features)
        x = x.view(chunk_size, batch_size * S, num_features) # (chunk_size, batch_size, S, num_features) -> (chunk_size, batch_size * S, num_features)
        embedding = self.positional_encoding(x)
        x = x + embedding
        x = self.transformer(x) # (chunk_size, batch_size * S, num_features)
        x = x.view(chunk_size, batch_size, S, num_features) # (chunk_size, batch_size * S, num_features) -> (chunk_size, batch_size, S, num_features)
        x = x.permute(1, 3, 2, 0).contiguous() # (chunk_size, batch_size, S, num_features) -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output


class InterTransformer(nn.Module):
    def __init__(self, num_features, num_layers=8, num_heads=8, d_ff=1024, norm=True, nonlinear='relu', dropout=1e-1, causal=False, norm_first=False, eps=EPS):
        super().__init__()

        self.num_features = num_features

        if isinstance(norm, int):
            if norm:
                norm_name = 'cLN' if causal else 'gLN'
                layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)
            else:
                layer_norm = None
        else:
            norm_name = norm
            layer_norm = LayerNormWrapper(norm_name, num_features, causal=False, batch_first=False, eps=eps)

        self.positional_encoding = PositionalEncoding(num_features, batch_first=False)
        encoder_layer = nn.TransformerEncoderLayer(num_features, num_heads, d_ff, dropout=dropout, activation=nonlinear, layer_norm_eps=eps, batch_first=False, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        residual = input
        x = input.permute(2, 0, 3, 1).contiguous() # (batch_size, num_features, S, chunk_size) -> (S, batch_size, chunk_size, num_features)
        x = x.view(S, batch_size * chunk_size, num_features) # (S, batch_size, chunk_size, num_features) -> (S, batch_size * chunk_size, num_features)
        embedding = self.positional_encoding(x)
        x = x + embedding
        # print('foward', x.shape)
        x = self.transformer(x) # (S, batch_size*chunk_size, num_features)
        x = x.view(S, batch_size, chunk_size, num_features) # (S, batch_size * chunk_size, num_features) -> (S, batch_size, chunk_size, num_features)
        x = x.permute(1, 3, 0, 2) # (S, batch_size, chunk_size, num_features) -> (batch_size, num_features, S, chunk_size)
        output = x + residual

        return output


class LayerNormWrapper(nn.Module):
    def __init__(self, norm_name, num_features, causal=False, batch_first=False, eps=EPS):
        super().__init__()

        self.batch_first = batch_first

        if norm_name in ['BN', 'batch', 'batch_norm']:
            kwargs = {
                'n_dims': 1
            }
        else:
            kwargs = {}

        self.norm1d = choose_layer_norm(norm_name, num_features, causal=causal, eps=eps, **kwargs)

    def forward(self, input):
        """
        Args:
            input: (T, batch_size, num_features) or (batch_size, T, num_features) if batch_first
        Returns:
            output: (T, batch_size, num_features) or (batch_size, T, num_features) if batch_first
        """
        if self.batch_first:
            input = input.permute(0, 2, 1).contiguous()
        else:
            input = input.permute(1, 2, 0).contiguous()

        output = self.norm1d(input)

        if self.batch_first:
            output = output.permute(0, 2, 1).contiguous()
        else:
            output = output.permute(2, 0, 1).contiguous()

        return output
    

def choose_layer_norm(name, num_features, causal=False, eps=EPS, **kwargs):
    if name == 'cLN':
        layer_norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif name == 'gLN':
        if causal:
            raise ValueError("Global Layer Normalization is NOT causal.")
        layer_norm = GlobalLayerNorm(num_features, eps=eps)
    elif name in ['BN', 'batch', 'batch_norm']:
        n_dims = kwargs.get('n_dims') or 1
        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError("n_dims is expected 1 or 2, but give {}.".format(n_dims))
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm



EPS = 1e-12

class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, *)
        Returns:
            output (batch_size, C, *)
        """
        output = self.norm(input)

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.fill_(0)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T) or (batch_size, C, S, chunk_size):
        Returns:
            output (batch_size, C, T) or (batch_size, C, S, chunk_size): same shape as the input
        """
        eps = self.eps

        n_dims = input.dim()

        if n_dims == 3:
            batch_size, C, T = input.size()
        elif n_dims == 4:
            batch_size, C, S, chunk_size = input.size()
            T = S * chunk_size
            input = input.view(batch_size, C, T)
        else:
            raise ValueError("Only support 3D or 4D input, but given {}D".format(input.dim()))

        step_sum = torch.sum(input, dim=1) # (batch_size, T)
        step_squared_sum = torch.sum(input**2, dim=1) # (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # (batch_size, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1) # (batch_size, T)

        cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float) # (T, ): [C, 2*C, ..., T*C]
        cum_mean = cum_sum / cum_num # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean**2

        cum_mean, cum_var = cum_mean.unsqueeze(dim=1), cum_var.unsqueeze(dim=1)

        output = (input - cum_mean) / (torch.sqrt(cum_var) + eps) * self.gamma + self.beta

        if n_dims == 4:
            output = output.view(batch_size, C, S, chunk_size)

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)
    

def _test_sepformer_block():
    d_model, d_ff = 256, 1024
    input = torch.randn((1, d_model, 250, 130))
    print('input shape', input.shape)
    model = SepFormerBlock(d_intra=d_model, d_inter=d_model, d_ff_intra=d_ff, d_ff_inter=d_ff)
    output = model(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    print("="*10, "SepFormerBlock", "="*10)
    _test_sepformer_block()
    print()