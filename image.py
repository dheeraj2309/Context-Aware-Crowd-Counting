import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import torch # We'll use torch for some ops

def draw_fixed_gaussian(heatmap, center, sigma):
    """Draws a 2D Gaussian blob with a fixed sigma."""
    # Get the integer center coordinates
    x, y = int(center[0]), int(center[1])
    
    h, w = heatmap.shape
    
    # Create a grid of coordinates
    ul = [int(x - 3 * sigma), int(y - 3 * sigma)]
    br = [int(x + 3 * sigma + 1), int(y + 3 * sigma + 1)]
    
    # Ensure the grid is within the heatmap bounds
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
        
    size = 2 * int(3 * sigma) + 1
    x_grid = np.arange(0, size, 1, float)
    y_grid = x_grid[:, np.newaxis]
    
    x0 = y0 = size // 2
    # The gaussian blob, not normalized
    g = np.exp(- ((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))

    # Define the paste location on the main heatmap
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    
    # Paste the gaussian blob, taking the maximum value if there's an overlap
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        
    return heatmap

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
    
    # --- TUNE THIS VALUE ---
    # This is the fixed sigma for the Gaussian, applied AT THE OUTPUT SCALE.
    # If you used sigma=30 on the full-size image, the equivalent on the 1/8th scale map
    # would be 30 / 8 = 3.75. Let's start with a value around there.
    FIXED_SIGMA = 4.0 
    # -----------------------

    output_w = img.width // output_stride
    output_h = img.height // output_stride

    gt_heatmap = np.zeros((output_h, output_w), dtype=np.float32)
    gt_bbox = np.zeros((4, output_h, output_w), dtype=np.float32)
    gt_offset = np.zeros((2, output_h, output_w), dtype=np.float32)
    gt_reg_mask = np.zeros((output_h, output_w), dtype=np.uint8)

    for index, row in annotations.iterrows():
        box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        
        if box_w <= 0 or box_h <= 0:
            continue

        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        center_out_x = center_x / output_stride
        center_out_y = center_y / output_stride
        
        int_center_out_x = int(center_out_x)
        int_center_out_y = int(center_out_y)

        # --- MODIFIED: Use the new fixed drawing function ---
        # No more adaptive radius calculation.
        if 0 <= int_center_out_x < output_w and 0 <= int_center_out_y < output_h:
            draw_fixed_gaussian(gt_heatmap, (center_out_x, center_out_y), FIXED_SIGMA)
        
            # The rest of the GT generation is the same
            gt_bbox[0, int_center_out_y, int_center_out_x] = box_w / img.width
            gt_bbox[1, int_center_out_y, int_center_out_x] = box_h / img.height
            gt_bbox[2, int_center_out_y, int_center_out_x] = center_x / img.width
            gt_bbox[3, int_center_out_y, int_center_out_x] = center_y / img.height

            gt_offset[0, int_center_out_y, int_center_out_x] = center_out_x - int_center_out_x
            gt_offset[1, int_center_out_y, int_center_out_x] = center_out_y - int_center_out_y

            gt_reg_mask[int_center_out_y, int_center_out_x] = 1
        
    return img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask