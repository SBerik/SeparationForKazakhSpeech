from .conv_tasnet import Conv_TasNet
from .dualpathrnn import Dual_RNN_model as DualPath_RNN
from .sepformer import Sepformer
from .superior_sepformer import SuperiorSepformer
from .pl_dualpathrnn import PL_Dual_RNN_model
from .pl_sepformer import PL_SuperiorSepformer


MODELS = {
    "Conv_TasNet": Conv_TasNet,
    "DualPath_RNN": DualPath_RNN,
    "Sepformer": Sepformer,
    "SuperiorSepformer": SuperiorSepformer, 
    "PL_Dual_RNN_model": PL_Dual_RNN_model,
    "PL_SuperiorSepformer": PL_SuperiorSepformer
}

def get_model(model_name):
    valid_model_names = [
    'DualPath_RNN',
    'Conv_TasNet',
    'Sepformer',
    'SuperiorSepformer',
    'PL_Dual_RNN_model',
    'PL_SuperiorSepformer'
    ]
    assert model_name in valid_model_names, 'Invalid model name'
    return MODELS[model_name]