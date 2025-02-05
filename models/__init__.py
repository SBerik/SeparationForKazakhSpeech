from .conv_tasnet import Conv_TasNet
from .dualpathrnn import Dual_RNN_model as DualPath_RNN
from .sepformer import Sepformer
from .superior_sepformer import SuperiorSepformer
from .pl_dualpathrnn import PL_Dual_RNN_model


MODELS = {
    "Conv_TasNet": Conv_TasNet,
    "DualPath_RNN": DualPath_RNN,
    "Sepformer": Sepformer,
    "SuperiorSepformer": SuperiorSepformer, 
    "PL_Dual_RNN_model": PL_Dual_RNN_model
}

def get_model(model_name):
    valid_model_names = [
    'DualPath_RNN',
    'Conv_TasNet',
    'Sepformer',
    'SuperiorSepformer',
    'PL_Dual_RNN_model'
    ]
    assert model_name in valid_model_names, 'Invalid model name'
    return MODELS[model_name]