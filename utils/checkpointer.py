import torch

class Checkpointer:
    def __init__(self, model_name, path_to_weights, 
                 checkpoint_path, metrics = True):
        self.model_name = model_name
        self.path_to_weights = path_to_weights
        self.checkpoint_path = checkpoint_path
        self.metrics = metrics 

    def save_best_weight(self, model, optimizer, epoch, epoch_state): 
        val_loss = epoch_state['valid']['loss'] 
        val_acc = epoch_state['valid']['metrics']['Accuracy'] if self.metrics else '_'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_state': epoch_state
        }, '{}/{}_{}_{:.4f}_{:.4f}.pt'.format(
                                              self.path_to_weights, self.model_name, 
                                              epoch, val_loss, val_acc
                                              ))

    def save_checkpoint(self, model, optimizer, epoch, epoch_state):
        val_loss = epoch_state['valid']['loss']
        val_acc = epoch_state['valid']['metrics']['Accuracy'] if self.metrics else '_'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_metrics': epoch_state
        }, '{}/checkpoint_{}_epoch_{}.pt'.format(
                                                self.checkpoint_path, self.model_name, epoch))