## Multi-Task Vehicle Detection and Counting  
This project implements a multi-task deep learning model for simultaneous vehicle detection (bounding boxes) and counting (density maps). It extends the popular Context-Aware Crowd Counting (CANNet) model by adding parallel heads for bounding box and center offset regression, creating a powerful tool for comprehensive traffic analysis.  
The model uses a shared VGG-16 backbone and then splits into two main branches:  
**Density Branch**: The original CANNet architecture is used to predict a density heatmap, which is excellent for counting objects in crowded scenes.  
**Detection Branch**: New convolutional layers are added to regress bounding box dimensions (width, height, center_x, center_y) and sub-pixel center offsets for each detected vehicle.  
This multi-task approach allows the model to leverage shared features for both tasks, leading to robust and accurate predictions.  
### Features  
Multi-Task Learning: Simultaneously predicts density maps for counting and bounding boxes for localization.  
CANNet Backbone: Leverages the powerful context-aware modules of CANNet for robust feature extraction.  
Advanced Loss Functions: Utilizes a combination of Focal Loss for the density heatmap (to handle class imbalance between object centers and background) and DIoU Loss for stable and accurate bounding box regression.  
Transfer Learning: Supports loading weights from a pre-trained CANNet model to kick-start training of the density branch.  
Flexible Training: The training script includes:  
* Differential learning rates for the backbone and new heads.  
* Resuming from checkpoints.  
* Options to freeze backbone layers.  
* Early stopping to prevent overfitting.  
* Detailed logging of training and validation metrics.  
* Data Augmentation: Includes random cropping and horizontal flipping during training to improve model generalization.  
### Project Structure
├── vehicle_model.py    # Defines the main VehicleDetector multi-task model.  
├── model.py            # Defines the original CANNet architecture (used as a building block).  
├── train.py            # Main script to train the multi-task model.  
├── test.py             # Script to evaluate a trained model (originally for CANNet).  
├── losses.py           # Defines MultiTaskLoss, FocalLoss, and DIoU Loss.  
├── image.py            # Data loading, augmentation, and ground truth generation.  
├── generate_heatmaps.py# Pre-processes point annotations into density map ground truth.  
├── utils.py            # Utility functions, including checkpoint saving.  
├── requirements.txt    # Python package dependencies.  
└── README.md           # This file.  
### Setup and Installation  
* Clone the Repository  
```bash
git clone https://github.com/dheeraj2309/Context-Aware-Crowd-Counting
cd Context-Aware-Crowd-Counting
```
* Create a Virtual Environment (Recommended)
``` bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
* Install Dependencies
The project requires several Python libraries. Install them using pip:
```bash
pip install -r requirements.txt
```

*If a `requirements.txt` file is not available, you can create one with the following content*:
```bash
torch
torchvision
numpy
pandas
h5py
scipy
Pillow
matplotlib
tqdm
scikit-learn
opencv-python
```

### Dataset Preparation
The model requires a specific data structure and annotations.  
1. **Directory Structure**  
Organize your dataset as follows. The script `generate_heatmaps.py` is configured to work with this structure.  
```bash
<your_data_root>/
├── train/
│   ├── images/
│   │   ├── image_0001.jpg
│   │   └── ...
│   └── ground_truth/
│       ├── GT_image_0001.mat
│       └── ...
└── valid/
    ├── images/
    │   ├── image_0100.jpg
    │   └── ...
    └── ground_truth/
        ├── GT_image_0100.mat
        └── ...
```
2. **Annotation Format**
The model uses two types of annotations:  
`.mat` files: For point-level annotations used to generate the initial density maps. Each .mat file should correspond to an image and contain a structure like image_info[0,0]['location'][0,0] which holds an Nx2 array of (x, y) coordinates.  
`.csv` file: A master annotation file containing bounding box information for all images. This is used by the training script to generate ground truth for the detection head. The CSV must have at least these columns: image_path, xmin, ymin, xmax, ymax.  
3. **Generate Ground Truth Density Maps**
The script `generate_heatmaps.py` converts the point annotations in the `.mat` files into Gaussian density maps and saves them as `.h5` files in the ground_truth directory.
* First, visualize to find the best GAUSSIAN_SIGMA:  
* Open `generate_heatmaps.py.`  
* Set ENABLE_VISUALIZATION = True.  
* Adjust GAUSSIAN_SIGMA until the generated heatmaps visually correspond well to the object sizes in the sample images.  
```bash
python generate_heatmaps.py
```
* After visualization, you will be prompted to proceed with generating all maps.  
Then, generate all .h5 files:  
Set ENABLE_VISUALIZATION = False and run the script again, or type yes at the prompt.  
4. **Prepare JSON Splits**
Create `train.json` and `val.json files`. These files should contain a list of absolute or relative paths to the image files for the training and validation sets, respectively.  
Example `train.json`:
 ```json
[
  "/path/to/your_data_root/train/images/image_0001.jpg",
  "/path/to/your_data_root/train/images/image_0002.jpg"
]
```
### Usage
* Training the Model
The `train.py` script is the main entry point for training. It uses argparse for configuration.  
Basic Training from Scratch:  
```bash
python train.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --annotations_csv /path/to/master_annotations.csv \
    --save_path ./checkpoints/run1 \
    --batch_size 4 \
    --epochs 100
```
* Training with Pre-trained CANNet Weights (Transfer Learning):
This is highly recommended. First, obtain a pre-trained CANNet model (.pth.tar file).
```bash
python train.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --annotations_csv /path/to/master_annotations.csv \
    --save_path ./checkpoints/run_transfer \
    --pretrained_cannet /path/to/pretrained_cannet.pth.tar \
    --lr_backbone 1e-5 \
    --lr_heads 1e-4
```
This sets a lower learning rate for the CANNet backbone and a higher one for the newly initialized detection heads.  
Resuming an Interrupted Training Session:  
The script automatically saves `latest_checkpoint.pth.tar` in the --save_path directory.  
```bash
python train.py \
    --train_json /path/to/train.json \
    --val_json /path/to/val.json \
    --annotations_csv /path/to/master_annotations.csv \
    --save_path ./checkpoints/run_transfer \
    --resume ./checkpoints/run_transfer/latest_checkpoint.pth.tar
```
### Evaluation
During training, the validate function calculates validation loss and Mean Absolute Error (MAE) for the count.
The provided test.py script is designed for the original CANNet model (it sums the density map to get a count). To evaluate the full multi-task model, you would need to write a new script that performs the following steps on the model's output:
* Apply sigmoid to the density map output.
* Perform non-maximum suppression (or simple peak finding) on the density map to identify object centers.
* For each identified center, retrieve the corresponding bounding box and offset predictions.
* Combine the predictions to generate final bounding boxes.
* Compare these boxes against ground truth using metrics like AP (Average Precision).
### Model Architecture Explained
The VehicleDetector model in `vehicle_model.py` works as follows:
**Shared Backbone**: An input image is passed through the cannet.frontend (a VGG-16 feature extractor), producing a feature map of size (B, 512, H/8, W/8).  
**Branch 1**: Density Prediction (CANNet Path)
* The shared features are fed into the standard CANNet context and backend modules.
* This branch outputs a single-channel density_map of size (B, 1, H/8, W/8).  
**Branch 2**: Detection Head Base  
* The same shared features are fed into a new, parallel convolutional block (new_heads_base).
* This produces a separate feature map (new_head_features) of size (B, 128, H/8, W/8).
### Feature Fusion and Final Prediction
A key architectural choice is to concatenate the output of the new detection branch with the output of the density branch: `torch.cat([new_head_features, density_map], dim=1)`  
This 129-channel feature map is then passed through two final 1x1 convolutional layers:  
* **bbox_head**: Predicts 4 channels for bounding box parameters.
* **offset_head**: Predicts 2 channels for sub-pixel center offsets.  
This design allows the final bounding box and offset predictions to be conditioned on the model's density estimation, potentially improving localization accuracy.
