import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import torch # We'll use torch for some ops

def gaussian_radius(det_size, min_overlap=0.7):
    """Calculate the radius of the 2D Gaussian heatmap.
    This is a standard formula used in center-point based detectors.
    """
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D Gaussian blob on a heatmap."""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def load_data_multi_head(img_path, annotations, train=True):
    """
    Loads an image and generates ground truth for density, bbox, and offset heads.
    'annotations' is a pandas DataFrame filtered for the current img_path.
    """
    img = Image.open(img_path).convert('RGB')
    
    if train:
        # --- Geometric Augmentations ---
        ratio = 0.5 
        crop_w = int(img.size[0] * ratio)
        crop_h = int(img.size[1] * ratio)
        dx = (img.size[0] - crop_w) if random.random() > 0.5 else 0
        dy = (img.size[1] - crop_h) if random.random() > 0.5 else 0

        # Update annotations to be relative to the crop
        annotations['xmin'] = annotations['xmin'] - dx
        annotations['ymin'] = annotations['ymin'] - dy
        annotations['xmax'] = annotations['xmax'] - dx
        annotations['ymax'] = annotations['ymax'] - dy
        
        # --- THIS IS THE CRITICAL SECTION ---
        # Filter out boxes that are no longer in the cropped image
        filtered_annotations = annotations[
            (annotations['xmax'] > 0) & (annotations['ymax'] > 0) &
            (annotations['xmin'] < crop_w) & (annotations['ymin'] < crop_h)
        ]
        
        # FIX: Explicitly create a copy after filtering to resolve the warning.
        annotations = filtered_annotations.copy()
        # ------------------------------------

        # Now, all subsequent modifications are safely performed on a definite copy.
        # Clamp boxes to be within the crop dimensions
        # We can use .loc to be even more explicit, as the warning suggests.
        annotations.loc[:, 'xmin'] = np.maximum(annotations['xmin'], 0)
        annotations.loc[:, 'ymin'] = np.maximum(annotations['ymin'], 0)
        annotations.loc[:, 'xmax'] = np.minimum(annotations['xmax'], crop_w)
        annotations.loc[:, 'ymax'] = np.minimum(annotations['ymax'], crop_h)
        
        # Perform the crop on the image
        img = img.crop((dx, dy, crop_w + dx, crop_h + dy))
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            old_x1 = annotations['xmin'].copy()
            old_x2 = annotations['xmax'].copy()
            annotations.loc[:, 'xmin'] = crop_w - old_x2
            annotations.loc[:, 'xmax'] = crop_w - old_x1

    # --- Ground Truth Generation ---
    output_stride = 8
    output_w = img.width // output_stride
    output_h = img.height // output_stride

    # Initialize empty ground truth maps
    gt_heatmap = np.zeros((output_h, output_w), dtype=np.float32)
    gt_bbox = np.zeros((4, output_h, output_w), dtype=np.float32)
    gt_offset = np.zeros((2, output_h, output_w), dtype=np.float32)
    # Regression mask to calculate loss only at object centers
    gt_reg_mask = np.zeros((output_h, output_w), dtype=np.uint8)

    for index, row in annotations.iterrows():
        # Get original box dimensions
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        box_w, box_h = x2 - x1, y2 - y1
        
        if box_w <= 0 or box_h <= 0:
            continue

        # Calculate center point and scale it to the output size
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        center_out_x = center_x / output_stride
        center_out_y = center_y / output_stride
        
        # Integer coordinates of the center on the output map
        int_center_out_x = int(center_out_x)
        int_center_out_y = int(center_out_y)

        # 1. Generate Density/Center-ness Heatmap
        # The radius of the gaussian is adaptive to the object's size
        radius = gaussian_radius((box_h, box_w))
        radius = max(0, int(radius / output_stride))
        draw_umich_gaussian(gt_heatmap, (center_out_x, center_out_y), radius)

        # 2. Generate BBox Target Map
        # We predict the size (w,h) of the box. Normalize by image size for stable training.
        # This could also be (l,t,r,b) offsets. Let's start with w,h.
        gt_bbox[0, int_center_out_y, int_center_out_x] = box_w / img.width
        gt_bbox[1, int_center_out_y, int_center_out_x] = box_h / img.height
        # Let's add the center as well for a DIoU loss later
        gt_bbox[2, int_center_out_y, int_center_out_x] = center_x / img.width
        gt_bbox[3, int_center_out_y, int_center_out_x] = center_y / img.height

        # 3. Generate Offset Target Map
        # The offset is the fractional part of the center coordinate
        dx = center_out_x - int_center_out_x
        dy = center_out_y - int_center_out_y
        gt_offset[0, int_center_out_y, int_center_out_x] = dx
        gt_offset[1, int_center_out_y, int_center_out_x] = dy

        # 4. Set the regression mask for this object's location
        gt_reg_mask[int_center_out_y, int_center_out_x] = 1
        
    return img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask