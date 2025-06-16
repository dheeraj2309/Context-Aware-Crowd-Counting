import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms # Import the main transforms module

# Import our new data loading function from image.py
from image import load_data_multi_head 

class MultiHeadDataset(Dataset):
    def __init__(self, json_path, annotation_csv_path, train=False):
        
        # Load the list of image paths from the JSON file
        with open(json_path, 'r') as f:
            self.img_paths = json.load(f)
        
        # Load the entire annotation CSV into memory
        self.annotations = pd.read_csv(annotation_csv_path)
        # Create a basename column for easy matching
        self.annotations['basename'] = self.annotations['filename'].apply(os.path.basename)
        # Group annotations by basename for fast lookup
        self.annotations_grouped = self.annotations.groupby('basename')
        
        self.train = train

        # --- AUGMENTATION PIPELINE ---
        # Define separate transformation pipelines for training and validation.
        
        # For training, we apply aggressive color augmentations.
        # Geometric augmentations (crop, flip) are handled inside load_data_multi_head.
        if self.train:
            self.photometric_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.25, # Randomly change brightness by up to 25%
                    contrast=0.25,   # Randomly change contrast by up to 25%
                    saturation=0.25, # Randomly change saturation by up to 25%
                    hue=0.1          # Randomly change hue by up to 10%
                )
                # You could add other transforms here, like GaussianBlur:
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            ])
        
        # For both training and validation, we need to convert to tensor and normalize.
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.train:
            random.shuffle(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        basename = os.path.basename(img_path)

        try:
            # Get all annotations for this specific image
            img_annotations = self.annotations_grouped.get_group(basename).copy()
        except KeyError:
            # Create an empty dataframe if an image has no annotations
            img_annotations = pd.DataFrame(columns=self.annotations.columns)

        # Generate all ground truth maps and perform geometric augmentations
        img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask = load_data_multi_head(
            img_path, img_annotations, self.train
        )

        # --- Apply the appropriate transformations ---
        # If in training mode, apply the color augmentations first.
        if self.train:
            img = self.photometric_transform(img)

        # Apply the final conversion to a normalized tensor for both train and val.
        img = self.tensor_transform(img)
            
        # Convert numpy ground truth maps to tensors
        gt_heatmap = torch.from_numpy(gt_heatmap)
        gt_bbox = torch.from_numpy(gt_bbox)
        gt_offset = torch.from_numpy(gt_offset)
        gt_reg_mask = torch.from_numpy(gt_reg_mask)

        return img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask