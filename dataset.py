
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import load_data # Import from the file we just created

class listDataset(Dataset):
    def __init__(self, root, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        # The 'root' is now a list of image paths from a json file
        if shuffle:
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img_path = self.lines[index]
        
        img, target = load_data(img_path, self.train)
        
        # Ensure target is writable for PyTorch
        target = np.copy(target)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target