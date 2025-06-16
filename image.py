
import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2

def load_data(img_path, train=True):
    # Construct the ground truth path by replacing 'images' with 'ground_truth'
    # and '.jpg' with '.h5'. This now points to our generated .h5 files.
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    
    img = Image.open(img_path).convert('RGB')
    
    try:
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])
    except FileNotFoundError:
        print(f"FATAL: Could not find ground truth file: {gt_path}")
        # In a real scenario, you might want to raise an exception or handle this gracefully
        return None, None

    # Apply data augmentation only during training
    if train:
        # The original code crops to a quarter of the image. This can be aggressive.
        # Let's keep it for consistency with the original training setup.
        ratio = 0.5 
        crop_size_w = int(img.size[0] * ratio)
        crop_size_h = int(img.size[1] * ratio)
        
        # Randomly select one of the four quarters
        if random.random() < 0.5:
            dx = 0
        else:
            dx = int(img.size[0] * ratio)
        
        if random.random() < 0.5:
            dy = 0
        else:
            dy = int(img.size[1] * ratio)
        
        img = img.crop((dx, dy, crop_size_w + dx, crop_size_h + dy))
        target = target[dy:crop_size_h + dy, dx:crop_size_w + dx]

        # Random horizontal flip
        if random.random() > 0.5: # Increased probability for more augmentation
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
    # Downsample the density map to 1/8th of the size, which is the model's output size.
    # The sum must be preserved, so we multiply by 64 (8*8).
    # We must ensure the target shape is divisible by 8.
    # Let's resize based on the augmented image size.
    target_h = target.shape[0] // 8
    target_w = target.shape[1] // 8
    
    target = cv2.resize(target, (target_w, target_h), interpolation=cv2.INTER_CUBIC) * 64

    return img, target