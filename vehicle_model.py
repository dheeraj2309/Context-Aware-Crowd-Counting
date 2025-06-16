import torch
import torch.nn as nn
# We import the original CANNet to use it as a building block
from model import CANNet, make_layers 

class VehicleDetector(nn.Module):
    def __init__(self):
        super(VehicleDetector, self).__init__()
        
        # 1. Instantiate the original CANNet model.
        # This will be our "density branch".
        self.cannet = CANNet()

        # 2. Define the new, parallel heads for BBox and Offset regression.
        # IMPORTANT: This new branch must take the same input as CANNet's context module,
        # which is the output of the `frontend`. The `frontend` produces 512 channels.
        
        # We'll create a small, shared convolutional block for our new heads.
        new_heads_shared_feat = [512, 256, 128] # A simple conv stack
        self.new_heads_base = make_layers(cfg=new_heads_shared_feat, in_channels=512, batch_norm=True, dilation=True)

        # Final 1x1 conv layers for each specific task
        # They take the 128-channel output from our new base block.
        self.bbox_head = nn.Conv2d(128, 4, kernel_size=1)   # 4 channels for box regression (e.g., l,t,r,b)
        self.offset_head = nn.Conv2d(128, 2, kernel_size=1) # 2 channels for center offset

    def forward(self, x):
        # 1. Pass the input through the shared VGG backbone (the frontend of CANNet)
        shared_features = self.cannet.frontend(x)

        # --- Branch 1: The Original CANNet Path for Density ---
        # Pass the shared features through the rest of the CANNet
        density_map = self.cannet.context(shared_features)
        density_map = self.cannet.backend(density_map)
        density_map = self.cannet.output_layer(density_map)
        
        # --- Branch 2: The New Heads Path for BBox/Offset ---
        # Pass the SAME shared features through our new head modules
        new_head_features = self.new_heads_base(shared_features)
        bbox_map = self.bbox_head(new_head_features)
        offset_map = self.offset_head(new_head_features)
        
        # The density map should be a probability, so apply a sigmoid.
        # The original CANNet doesn't do this, it outputs raw density values.
        # We'll keep it as raw density to match the original training.
        # If we use a Focal Loss later, we will apply sigmoid there.

        return density_map, bbox_map, offset_map

    def load_pretrained_cannet(self, checkpoint_path, device):
        """
        Loads the full state_dict from a pre-trained CANNet checkpoint
        into the `self.cannet` submodule of our new model.
        """
        print(f"=> Loading pre-trained weights into the CANNet branch from: {checkpoint_path}")
        
        # This will load the weights for frontend, context, backend, and output_layer
        # into our `self.cannet` component.
        self.cannet.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])

        print("=> Pre-trained CANNet weights loaded successfully.")
        print("=> The new BBox and Offset heads are randomly initialized.")