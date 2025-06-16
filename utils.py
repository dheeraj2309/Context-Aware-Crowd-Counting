import h5py
import torch
import shutil
import numpy as np
import os # Import os

# The save_net and load_net functions are fine, but are not used by the main training script.
# They are kept here for completeness.
def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)

def save_checkpoint(state, is_best, save_path):
    """
    Saves the training checkpoint.

    Args:
        state (dict): A dictionary containing model state, optimizer state, etc.
        is_best (bool): True if this is the best model seen so far.
        save_path (str): The directory where checkpoints will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Define the file paths
    latest_checkpoint_path = os.path.join(save_path, 'latest_checkpoint.pth.tar')
    best_model_path = os.path.join(save_path, 'model_best.pth.tar')
    
    # 1. Always save the latest checkpoint.
    # This file is continuously overwritten and is used for resuming interrupted runs.
    torch.save(state, latest_checkpoint_path)
    
    # 2. If this checkpoint is the best so far, save it as the best model.
    # This file is only overwritten when a new best validation score is achieved.
    if is_best:
        # We save the state again to a different file.
        # This is slightly more I/O but makes the logic very clear.
        torch.save(state, best_model_path)
        print(f"✅ Best model checkpoint saved to {best_model_path}")