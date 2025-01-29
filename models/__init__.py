from .conv_tasnet import Conv_TasNet
from .dualpathrnn import Dual_RNN_model as DualPath_RNN
from .sepformer import Sepformer
from .superior_sepformer import SuperiorSepformer


MODELS = {
    "Conv_TasNet": Conv_TasNet,
    "DualPath_RNN": DualPath_RNN,
    "Sepformer": Sepformer,
    "SuperiorSepformer": SuperiorSepformer
}


def get_model(model_name):
    assert model_name in ['DualPath_RNN', 'Conv_TasNet', 'Sepformer', 'SuperiorSepformer'], 'Invalid model name'
    return MODELS[model_name]