import sys
import os
import warnings
import time
import json
import argparse
import pandas as pd # Import pandas for logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

# Import your other project files
from model import CANNet
from utils import save_checkpoint
import dataset 

# --- Argument Parser (Unchanged) ---
parser = argparse.ArgumentParser(description='PyTorch CANNet Fine-Tuning with Logging')
# (Add all arguments from the previous script here)
# ...
parser.add_argument('train_json', metavar='TRAIN', help='path to train json')
parser.add_argument('val_json', metavar='VAL', help='path to val json')
parser.add_argument('--pretrained_model', type=str, default=None, help='path to pre-trained model for fine-tuning')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='path to save checkpoints and log')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--workers', type=int, default=2, help='data loading workers')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')


# --- Main Function (with logging added) ---
def main():
    global args, best_prec1
    best_prec1 = 1e6

    args = parser.parse_args()

    # --- Setup Logging ---
    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, 'training_log.csv')
    # If the log file doesn't exist, create it with a header
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write('epoch,lr,avg_train_loss,val_mae\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.train_json, 'r') as f: train_list = json.load(f)
    with open(args.val_json, 'r') as f: val_list = json.load(f)

    model = CANNet().to(device)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)

    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        print(f"=> loading checkpoint '{args.pretrained_model}'")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint successfully")
    else:
        print("=> No pre-trained model specified or found. Training from scratch.")

    # --- Training Loop (with logging added) ---
    for epoch in range(args.epochs):
        # The train function now returns the average loss
        avg_train_loss = train(train_list, model, criterion, optimizer, epoch, device)
        
        # The validate function returns the validation MAE
        val_mae = validate(val_list, model, criterion, device)

        is_best = val_mae < best_prec1
        best_prec1 = min(val_mae, best_prec1)
        print(f' * Best MAE so far {best_prec1:.3f}')
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_path, 'checkpoint.pth.tar'))
        
        # --- Log the statistics for this epoch ---
        current_lr = optimizer.param_groups[0]['lr']
        log_stats = f'{epoch},{current_lr},{avg_train_loss},{val_mae}\n'
        with open(log_file_path, 'a') as f:
            f.write(log_stats)

# --- AverageMeter Class (Unchanged) ---
class AverageMeter(object):
    # ... (code from previous response)
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- `train` function (modified to return avg_loss) ---
def train(train_list, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list, shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       ]), 
                       train=True, batch_size=args.batch_size, num_workers=args.workers),
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
            
    # Return the average loss for the entire epoch
    return losses.avg

# --- `validate` function (Unchanged, already returns MAE) ---
def validate(val_list, model, criterion, device):
    # ... (code from previous response)
    print('begin val')
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list, shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       ]), train=False),
        batch_size=1)
    model.eval()
    mae = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(val_loader):
            img = img.to(device)
            output = model(img)
            pred_count = output.sum().item()
            gt_count = target.sum().item()
            mae += abs(pred_count - gt_count)
    mae = mae / len(val_loader)    
    print(f' * MAE {mae:.3f}')
    return mae

if __name__ == '__main__':
    main()