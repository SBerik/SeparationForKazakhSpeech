import torch
from typing import List


def configure_optimizer(cfg, model):
    assert cfg['training']["optim"] in ['Adam', 'SGD'], "Invalid optimizer type"
    return (torch.optim.Adam if cfg['training']["optim"] == 'Adam' else torch.optim.SGD) (model.parameters(), 
                 lr=cfg['training']["lr"], weight_decay=cfg['training']["weight_decay"])


def torch_logger (writer, epoch, epoch_state):
    writer.add_scalars('Loss', {
        'Train': epoch_state['train']['loss'], 
        'Validation': epoch_state['valid']['loss']
    }, epoch)

    if epoch_state.metrics:
        for m in epoch_state['metrics_name']:
            writer.add_scalars(f'{m}', {
                'Train': epoch_state['train']['metrics'][m], 
                'Validation': epoch_state['valid']['metrics'][m]
            }, epoch)


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
    print("Size of model: {:.2f} MB, in {}".format(model_size, dtype))


def tensify(sample: List[torch.Tensor], device) -> torch.Tensor:
    return torch.stack(sample, dim=0).to(device)


class EpochState(dict):
    """api:
        from losses import sisnr_pit
        from torchmetrics.audio import PermutationInvariantTraining
        from torchmetrics.functional.audio import signal_distortion_ratio

        losses = {"sisnr": sisnr_pit, 
                "sdr": PermutationInvariantTraining(signal_distortion_ratio, 
                                                            mode="speaker-wise", 
                                                            eval_func="max")}
        epoch_losses = EpochState(losses)
        outputs, labels = ...
        sdr_i, sisnr_i = - losses["sdr"](), losses["sisnr"]()
        loss = aplha * sisnr_i + beta * sdr_i
        ...
        loss.backward()
        epoch_losses.add(loss, phase, {"sisnr": sisnr_i, "sdr": sdr_i})
        # running_loss += loss.item()
    # epoch_loss = running_loss / len(dataloader)
    epoch_losses.update_state(phase)
    add - что бы добавить (в случае loss), в случае метрик - обновить
    update - в случае метрик обновить и подсчетать
    """
    def __init__(self, metrics=None, epochs = 'cpu'):
        super().__init__()
        self.metrics = metrics
        if metrics:
            self['metrics_name'] = list(metrics.keys())
            for phase in ['train', 'valid']: 
                self[phase] = {'loss': 0.0, 'metrics': {m: 0.0 for m in self['metrics_name']}}
        else:
            for phase in ['train', 'valid']: 
                self[phase] = {'loss': 0.0}

    def update_loss(self, phase, loss):
        self[phase]['loss'] += loss.item()

    def compute_loss(self, phase, N):
        if phase not in self:
            raise KeyError(f"Phase '{phase}' not initialized in EpochState.")
        epoch_loss = self[phase]['loss'] / N
        self[phase]['loss'] = epoch_loss
        return epoch_loss

    def update_metrics(self, phase, metrics):
        """
        Todo: updating for 'torchmetrics' 
        """
        for name, value in metrics.items():
            self[phase]['metrics'][name] += value.item()

    def compute_metrics(self, phase, N):
        if phase not in self:
            raise KeyError(f"Phase '{phase}' not initialized in EpochState.")
        self[phase]['metrics'] = {k: val / N for k, val in self[phase]['metrics'].items()}
        return self[phase]['metrics']

    def p_output(self, epoch, phase):
        if phase == 'train':
            print(f'Epoch {epoch+1}/{self.epochs}')
        print(f"{phase.upper()}, Loss: {self[phase]['loss']:.4f}", end="")
        if self.metrics:
            for m in self['metrics_name']:
                print(f"{m}: {self[phase]['metrics'][m]:.4f} ", end="")
        print() 
        if phase == 'valid':
            print('-' * 108, '\n')