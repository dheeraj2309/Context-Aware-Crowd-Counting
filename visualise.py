import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import os
from PIL import Image
from pathlib import Path

# Import our model definition
from vehicle_model import VehicleDetector

# --- CONFIGURATION ---
# Path to the best saved model from your training run
MODEL_PATH = '/kaggle/working/vehicle_detector_run1/model_best.pth.tar'
# Path to your validation images folder
IMAGE_DIR = '/kaggle/input/vehicle-data/valid/images/'
# Number of random images to visualize
NUM_IMAGES = 5
# Confidence threshold for detecting peaks in the heatmap
CONF_THRESHOLD = 0.4

# --- Helper Functions for Post-Processing ---
def find_peaks(heatmap):
    """Finds local maxima in a heatmap that are above a threshold."""
    # Use a max filter to find local maxima
    heatmap_max = nn.functional.max_pool2d(heatmap, 3, stride=1, padding=1)
    peaks = (heatmap_max == heatmap) & (heatmap > CONF_THRESHOLD)
    peaks = peaks.squeeze() # Remove batch and channel dimensions
    
    # Get the coordinates of the peaks
    peak_coords = torch.nonzero(peaks, as_tuple=False)
    return peak_coords

def draw_predictions(image, heatmap, boxes, title=""):
    """Draws heatmap and bounding boxes on an image."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.set_title(title)
    
    # Overlay the heatmap
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    ax.imshow(heatmap_resized, cmap='jet', alpha=0.5) # Semi-transparent overlay

    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
    plt.show()

# --- Main Visualization Script ---
def visualize_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    model = VehicleDetector().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 2. Get a list of random image paths
    all_images = list(Path(IMAGE_DIR).glob('*.jpg'))
    sample_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

    # 3. Define image transformations (must match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Loop through images and make predictions
    for img_path in sample_images:
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_density, pred_bbox, pred_offset = model(img_tensor)

        # Move predictions to CPU and numpy for processing
        pred_density = torch.sigmoid(pred_density).cpu() # Apply sigmoid to get probabilities
        pred_bbox = pred_bbox.cpu().squeeze(0) # Remove batch dim
        pred_offset = pred_offset.cpu().squeeze(0)

        # 5. Post-process the output
        # Find the peaks (predicted centers) in the density map
        peaks = find_peaks(pred_density) # Returns [y, x] coordinates
        
        pred_count = len(peaks)
        print(f"Image: {img_path.name}, Predicted Count: {pred_count}")

        # Reconstruct bounding boxes from the output maps
        output_stride = 8
        final_boxes = []
        for y, x in peaks:
            # Get the bbox and offset predictions at the peak location
            box_pred = pred_bbox[:, y, x]
            offset_pred = pred_offset[:, y, x]

            # Reconstruct center coordinates with sub-pixel offset
            center_x = (x.float() + offset_pred[0]) * output_stride
            center_y = (y.float() + offset_pred[1]) * output_stride
            
            # De-normalize width and height
            width = box_pred[0] * original_img.width
            height = box_pred[1] * original_img.height

            # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            final_boxes.append((x1.item(), y1.item(), x2.item(), y2.item()))

        # 6. Draw the results
        heatmap_to_show = pred_density.squeeze().numpy()
        draw_predictions(original_img, heatmap_to_show, final_boxes, 
                         title=f"Prediction for {img_path.name} (Count: {pred_count})")

if __name__ == '__main__':
    # You would run this function in a notebook cell
    visualize_predictions()