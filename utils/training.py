import torch

def torch_logger (writer, phase, epoch, epoch_loss, epoch_metrics, metrics):
    writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
    [writer.add_scalar(f'{phase}/{m}', epoch_metrics[m], epoch) for m in metrics.keys()]


def p_output_log(epoch, num_epochs, phase, epoch_loss, epoch_metrics, metrics):
    if phase == 'train':
        print(f'Epoch {epoch+1}/{num_epochs}')
    print(f"{phase.upper()}, Loss: {epoch_loss:.4f}, ", end="")
    for m in metrics.keys():
        print(f"{m}: {epoch_metrics[m]:.4f} ", end="")
    print() 
    if phase == 'valid':
        print('-' * 108, '\n')


def save_best_weight(model, optimizer, epoch, epoch_loss, epoch_metrics, path_to_weights, model_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'epoch acc': epoch_metrics['Accuracy'],
        'epoch_metrics': epoch_metrics
    }, '{}/{}_{}_{:.4f}_{:.4f}.pt'.format(path_to_weights, model_name, epoch, epoch_loss, epoch_metrics['Accuracy']))


def save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_metrics, checkpoint_path, model_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'epoch acc': epoch_metrics['Accuracy'],
        'epoch_metrics': epoch_metrics
    }, '{}/checkpoint_{}_epoch_{}.pt'.format(checkpoint_path, model_name, epoch))


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