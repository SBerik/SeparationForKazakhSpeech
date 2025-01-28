from .conv_tasnet import Conv_TasNet
from .dualpathrnn import Dual_RNN_model as DualPath_RNN
from .sepformer import Sepformer
from .updated_sepformer import MySepfomer


MODELS = {
    "Conv_TasNet": Conv_TasNet,
    "DualPath_RNN": DualPath_RNN,
    "Sepformer": Sepformer,
    "MySepfomer": MySepfomer
}