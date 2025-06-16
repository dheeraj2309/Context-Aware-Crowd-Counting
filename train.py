import sys
import os
import warnings
import time
import json
import argparse # Import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

# These imports assume the files are in the same directory or accessible
from model import CANNet
from utils import save_checkpoint
import dataset 

# --- Argument Parser ---
# This section replaces the hardcoded configuration
parser = argparse.ArgumentParser(description='PyTorch CANNet Fine-Tuning')

# Path Arguments
parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='path to pre-trained model for fine-tuning')
parser.add_argument('--save_path', type=str, default='./checkpoints/',
                    help='path to save checkpoints')

# Hyperparameter Arguments
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate for fine-tuning')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size for training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--print_freq', type=int, default=20,
                    help='print frequency')

# --- Main Function ---
def main():
    global args, best_prec1
    best_prec1 = 1e6

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    with open(args.train_json, 'r') as f:
        train_list = json.load(f)
    with open(args.val_json, 'r') as f:
        val_list = json.load(f)

    # --- Model Definition ---
    model = CANNet().to(device)

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.decay)

    # --- Load Pre-trained Weights for Fine-Tuning ---
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print(f"=> loading checkpoint '{args.pretrained_model}'")
            checkpoint = torch.load(args.pretrained_model, map_location=device)
            # You might need to adjust this if the checkpoint structure is different
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.pretrained_model}' successfully")
        else:
            print(f"=> ERROR: No checkpoint found at '{args.pretrained_model}'")
            # Exit if the specified pre-trained model doesn't exist
            return
    else:
        print("=> No pre-trained model provided. Training from scratch.")

    # --- Training Loop ---
    for epoch in range(args.epochs):
        # Here you could implement a learning rate scheduler if needed
        # adjust_learning_rate(optimizer, epoch) 
        
        train(train_list, model, criterion, optimizer, epoch, device)
        prec1 = validate(val_list, model, criterion, device)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(f' * Best MAE so far {best_prec1:.3f}')
        
        # Save checkpoints to the specified path
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_path, 'checkpoint.pth.tar'))


# --- Helper classes and functions ---
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_list, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                       ]), 
                       train=True, 
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    
    print(f'Epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]:.5f}')
    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device)
        target = target.type(torch.FloatTensor).to(device)
        
        output = model(img)
        
        loss = criterion(output.squeeze(), target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(f'  [{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def validate(val_list, model, criterion, device):
    print('begin val')
    val_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                   ]), train=False),
    batch_size=1) # Validate one image at a time

    model.eval()
    mae = 0
    
    with torch.no_grad():
        for i,(img, target) in enumerate(val_loader):
            img = img.to(device)
            output = model(img)
            
            # Sum of the density map gives the count
            pred_count = output.sum().item()
            gt_count = target.sum().item()
            
            mae += abs(pred_count - gt_count)
            
    mae = mae / len(val_loader)    
    print(f' * MAE {mae:.3f}')
    return mae

if __name__ == '__main__':
    main()