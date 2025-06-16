import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a standard implementation of Focal Loss.
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        pred_sigmoid = torch.sigmoid(pred) # Apply sigmoid to raw logits
        pt = torch.where(torch.eq(gt, 1.), pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = torch.where(torch.eq(gt, 1.), self.alpha, 1 - self.alpha)
        focal_weight = focal_weight * (1 - pt) ** self.gamma
        
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none') * focal_weight
        return loss.sum()

# This is a standard implementation of DIoU Loss
def a_diou_loss(pred_boxes, gt_boxes):
    """
    Calculates the DIoU loss between predicted and ground truth boxes.
    pred_boxes: [N, 4] tensor of predicted boxes (cx, cy, w, h)
    gt_boxes: [N, 4] tensor of ground truth boxes (cx, cy, w, h)
    """
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    gx1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    gy1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gx2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    gy2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter_area = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

    # Union
    p_area = (px2 - px1) * (py2 - py1)
    g_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = p_area + g_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)

    # Bounding box of the two boxes
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    c_diag = (cx2 - cx1)**2 + (cy2 - cy1)**2 + 1e-6

    # Distance between centers
    dist = (pred_boxes[:, 0] - gt_boxes[:, 0])**2 + (pred_boxes[:, 1] - gt_boxes[:, 1])**2
    
    # DIoU Loss
    diou_loss = 1 - iou + (dist / c_diag)
    return diou_loss.sum()

class MultiTaskLoss(nn.Module):
    def __init__(self, density_weight=1.0, bbox_weight=5.0):
        super(MultiTaskLoss, self).__init__()
        self.density_loss_fn = FocalLoss()
        # We will use our DIoU loss function inside this class
        self.bbox_weight = bbox_weight
        self.density_weight = density_weight

    def forward(self, predictions, targets):
        pred_density, pred_bbox, pred_offset = predictions
        gt_heatmap, gt_bbox, gt_offset, gt_reg_mask = targets
        
        # --- 1. Density Head Loss (Focal Loss) ---
        # The pred_density from the model doesn't have a sigmoid, so the loss fn will apply it
        density_loss = self.density_loss_fn(pred_density.squeeze(1), gt_heatmap)
        
        # --- 2. BBox Head Loss (DIoU Loss) ---
        # We only calculate this loss where objects exist, using the regression mask.
        mask = gt_reg_mask.bool()
        
        if mask.sum() == 0:
            # No ground truth objects in this batch, so regression loss is 0
            bbox_loss = torch.tensor(0., device=pred_bbox.device)
        else:
            # Gather the predictions and ground truths only at the masked locations
            # pred_bbox shape is [B, 4, H, W], we want [N, 4] where N is num of objects
            pred_boxes_at_centers = pred_bbox.permute(0, 2, 3, 1)[mask]
            gt_boxes_at_centers = gt_bbox.permute(0, 2, 3, 1)[mask]

            bbox_loss = a_diou_loss(pred_boxes_at_centers, gt_boxes_at_centers)
            
        # Normalize losses by number of objects
        num_objects = mask.sum().float().clamp(min=1)
        density_loss = density_loss / num_objects
        bbox_loss = bbox_loss / num_objects

        # --- 3. Total Weighted Loss ---
        total_loss = (self.density_weight * density_loss) + (self.bbox_weight * bbox_loss)
        
        # Return individual losses for logging
        return total_loss, {'density_loss': density_loss, 'bbox_loss': bbox_loss}