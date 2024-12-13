import torch

def configure_optimizer(cfg, model):
    assert cfg['training']["optim"] in ['Adam', 'SGD'], "Invalid optimizer type"
    return (torch.optim.Adam if cfg['training']["optim"] == 'Adam' else torch.optim.SGD) (model.parameters(), 
                 lr=cfg['training']["lr"], weight_decay=cfg['training']["weight_decay"])


def torch_logger (writer, epoch, epoch_state):
    writer.add_scalars('Loss', {
        'Train': epoch_state['train']['loss'], 
        'Validation': epoch_state['valid']['loss']
    }, epoch)

    for m in epoch_state['metrics_name']:
        writer.add_scalars(f'{m}', {
            'Train': epoch_state['train']['metrics'][m], 
            'Validation': epoch_state['valid']['metrics'][m]
        }, epoch)


def p_output_log(num_epochs, epoch, phase, epoch_loss):
    if phase == 'train':
        print(f'Epoch {epoch+1}/{num_epochs}')
    print(f"{phase.upper()}, Loss: {epoch_loss:.4f}, ", end="")
    print() 
    if phase == 'valid':
        print('-' * 108, '\n')


def getNumParams (model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


def metadata_info (model, dtype = 'float32') -> None:
    num_params = getNumParams(model)
    if dtype == "float32":
        model_size = (num_params/1024**2) * 4
    elif dtype == "float16" or dtype == "bfloat16":
        model_size = (num_params/1024**2) * 2
    elif dtype == "int8":
        model_size = (num_params/1024**2) * 1
    else:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float32', 'float16', 'bfloat16', and 'int8'.")
    
    print(f"Trainable parametrs: {num_params}")
    print("Size of model: {:.2f} MB, in {}".format(model_size, dtype), '\n')


class EpochState(dict):
    '''
    Class: contains for current epoc losses and metrics 
    __state:
        metrics_name: [Acc, Pr, Fr]
        'train': {
                loss: 0.0,
                metrics:
                    'acc':
                    'pr':
                    'rc':
        }
        'valid': {
                'loss': 0.0
                metrics:
                    'acc':
                    'pr':
                    'rc':
        }
    '''
    def __init__(self, metrics):
        super().__init__()
        self['metrics_name'] = list(metrics.keys())
        for phase in ['train', 'valid']: 
            self[phase] = {'loss': float('inf'), 'metrics': {m: 0.0 for m in self['metrics_name']}}

    def update_state(self, loss, phase: str, metrics_val:dict):
        self[phase]['loss'] = loss
        self[phase]['metrics'] = metrics_val

# def torch_logger(writer, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
#     # Объединяем train и valid для Loss
#     writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
    
#     # Объединяем train и valid для Accuracy
#     writer.add_scalars('Accuracy',  {'Train': train_accuracy, 'Validation': val_accuracy}, epoch)
