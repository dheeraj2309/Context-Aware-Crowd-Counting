import sys
import os
import time
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F # Needed for the corrected MAE calculation

# Import all our custom modules
from vehicle_model import VehicleDetector
from dataset import MultiHeadDataset
from losses import MultiTaskLoss
from utils import save_checkpoint

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Vehicle Detector Training')
# Paths
parser.add_argument('--train_json', required=True, help='path to train json')
parser.add_argument('--val_json', required=True, help='path to val json')
parser.add_argument('--annotations_csv', required=True, help='path to the master annotations csv')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='path to save checkpoints and log')
# Loading Weights
parser.add_argument('--pretrained_cannet', type=str, default=None, help='path to pre-trained CANNet model for initial transfer learning')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint to resume training')
# Hyperparameters
parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for pre-trained parts')
parser.add_argument('--lr_heads', type=float, default=1e-4, help='learning rate for new heads')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', type=int, default=2, help='data loading workers')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
# Freezing and Early Stopping
parser.add_argument('--freeze_frontend', action='store_true', help='Freeze the VGG frontend')
parser.add_argument('--freeze_context', action='store_true', help='Freeze the CANNet context module')
parser.add_argument('--early_stop_patience', type=int, default=10, help='Patience for early stopping')

# --- Main Training Function ---
def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    
    # --- Setup Logging ---
    log_file_path = os.path.join(args.save_path, 'training_log.csv')
    # If not resuming, create a new log file
    if not args.resume:
        log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_density_loss', 'val_bbox_loss', 'val_mae'])
        log_df.to_csv(log_file_path, index=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = VehicleDetector().to(device)
    
    # --- Optimizer ---
    optimizer = torch.optim.Adam([
        {'params': model.cannet.frontend.parameters(), 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': model.cannet.context.parameters(), 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': model.cannet.backend.parameters(), 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': model.cannet.output_layer.parameters(), 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': model.new_heads_base.parameters(), 'lr': args.lr_heads, 'name': 'heads'},
        {'params': model.bbox_head.parameters(), 'lr': args.lr_heads, 'name': 'heads'},
        {'params': model.offset_head.parameters(), 'lr': args.lr_heads, 'name': 'heads'},
    ], weight_decay=args.decay)

    # --- Initialize variables for training loop ---
    start_epoch = 0
    best_val_loss = float('inf')

    # --- LOGIC FOR LOADING WEIGHTS ---
    # Priority: 1. Resume from a checkpoint, 2. Load pre-trained CANNet, 3. Train from scratch
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Resuming training from checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # --- THIS IS THE FULLY ROBUST FIX ---
            # Use .get() for ALL optional keys to prevent any KeyErrors.
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            model.load_state_dict(checkpoint['state_dict'])
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> Optimizer state loaded successfully.")
            else:
                print("=> WARNING: Optimizer state not found in checkpoint. Starting with a fresh optimizer.")

            print(f"=> Resumed from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
            # --- END OF FIX ---
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'")
            return
    elif args.pretrained_cannet:
        # This part runs only when starting a fresh training run with transfer learning
        model.load_pretrained_cannet(args.pretrained_cannet, device)

    # --- Layer Freezing Logic ---
    if args.freeze_frontend:
        print("Freezing frontend layers...")
        for param in model.cannet.frontend.parameters(): param.requires_grad = False
    if args.freeze_context:
        print("Freezing context module layers...")
        for param in model.cannet.context.parameters(): param.requires_grad = False

    # --- DataLoaders ---
    train_dataset = MultiHeadDataset(args.train_json, args.annotations_csv, train=True)
    val_dataset = MultiHeadDataset(args.val_json, args.annotations_csv, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    criterion = MultiTaskLoss().to(device)
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch}/{args.epochs-1} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_losses, val_mae = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.3f}")

        # Logging
        new_log = pd.DataFrame({
            'epoch': [epoch], 'train_loss': [train_loss], 'val_loss': [val_loss],
            'val_density_loss': [val_losses['density_loss']], 'val_bbox_loss': [val_losses['bbox_loss']], 'val_mae': [val_mae]
        })
        new_log.to_csv(log_file_path, mode='a', header=False, index=False)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"New best validation loss! Saving model to {args.save_path}")
        else:
            epochs_no_improve += 1
        
        # Always save the latest checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.save_path, 'checkpoint.pth.tar'))

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered after {args.early_stop_patience} epochs with no improvement.")
            break

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, (img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask) in enumerate(loader):
        img = img.to(device)
        targets = (gt_heatmap.to(device), gt_bbox.to(device), gt_offset.to(device), gt_reg_mask.to(device))
        
        optimizer.zero_grad()
        predictions = model(img)
        loss, _ = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f"  Batch {i}/{len(loader)}, Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae = 0, 0
    total_density_loss, total_bbox_loss = 0, 0
    
    # Peak finding needs nn.functional
    import torch.nn.functional as F

    with torch.no_grad():
        for img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask in loader:
            img = img.to(device)
            targets = (gt_heatmap.to(device), gt_bbox.to(device), gt_offset.to(device), gt_reg_mask.to(device))
            
            predictions = model(img)
            loss, loss_dict = criterion(predictions, targets)
            total_loss += loss.item()
            total_density_loss += loss_dict['density_loss'].item()
            total_bbox_loss += loss_dict['bbox_loss'].item()
            
            # --- CORRECTED MAE CALCULATION ---
            pred_heatmap = torch.sigmoid(predictions[0]) # Get heatmap probabilities

            # Use max pooling to find local maxima, just like in the visualization script
            h_max = F.max_pool2d(pred_heatmap, 3, stride=1, padding=1)
            # A peak is where the original value is a local max and above a threshold
            peaks = (h_max == pred_heatmap) & (pred_heatmap > 0.3) # Using a confidence threshold
            
            pred_count = peaks.sum().item() # The count is the number of peaks
            gt_count = gt_reg_mask.sum().item() # The true count
            total_mae += abs(pred_count - gt_count)
            # --- END OF CORRECTION ---

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader) # This will now be a meaningful number
    avg_losses = {'density_loss': total_density_loss / len(loader), 'bbox_loss': total_bbox_loss / len(loader)}

    return avg_loss, avg_losses, avg_mae

if __name__ == '__main__':
    # You will need to re-paste the full helper functions here to make the script self-contained
    # For example:
    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for i, (img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask) in enumerate(loader):
            img = img.to(device)
            targets = (gt_heatmap.to(device), gt_bbox.to(device), gt_offset.to(device), gt_reg_mask.to(device))
            optimizer.zero_grad()
            predictions = model(img)
            loss, _ = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(model, loader, criterion, device):
        model.eval()
        total_loss, total_mae = 0, 0
        total_density_loss, total_bbox_loss = 0, 0
        with torch.no_grad():
            for img, gt_heatmap, gt_bbox, gt_offset, gt_reg_mask in loader:
                img = img.to(device)
                targets = (gt_heatmap.to(device), gt_bbox.to(device), gt_offset.to(device), gt_reg_mask.to(device))
                predictions = model(img)
                loss, loss_dict = criterion(predictions, targets)
                total_loss += loss.item()
                total_density_loss += loss_dict['density_loss'].item()
                total_bbox_loss += loss_dict['bbox_loss'].item()
                pred_heatmap = torch.sigmoid(predictions[0])
                h_max = F.max_pool2d(pred_heatmap, 3, stride=1, padding=1)
                peaks = (h_max == pred_heatmap) & (pred_heatmap > 0.4)
                pred_count = peaks.sum().item()
                gt_count = gt_reg_mask.sum().item()
                total_mae += abs(pred_count - gt_count)
        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        avg_losses = {'density_loss': total_density_loss / len(loader), 'bbox_loss': total_bbox_loss / len(loader)}
        return avg_loss, avg_losses, avg_mae
        
    main()