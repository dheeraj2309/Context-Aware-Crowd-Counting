import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.scales]
        weights = [self.__make_weight(feats, scale_feature) for scale_feature in multi_scales]
        
        weights_sum = torch.stack(weights).sum(0)
        weights_sum[weights_sum == 0] = 1e-6

        weighted_features = (multi_scales[0] * weights[0] + multi_scales[1] * weights[1] + multi_scales[2] * weights[2] + multi_scales[3] * weights[3]) / weights_sum
        
        # The concatenation and bottleneck happen INSIDE this module
        bottle = self.bottleneck(torch.cat([weighted_features, feats], 1))
        return self.relu(bottle)

class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.seen = 0
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        # The backend correctly expects 512 channels because that's what self.context outputs
        self.backend = make_layers(self.backend_feat, in_channels=512, batch_norm=True, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # Use a more robust way to copy weights
            frontend_dict = self.frontend.state_dict()
            vgg_dict = mod.features.state_dict()
            # Filter out unnecessary keys
            vgg_dict = {k: v for k, v in vgg_dict.items() if k in frontend_dict}
            frontend_dict.update(vgg_dict)
            self.frontend.load_state_dict(frontend_dict)

    def forward(self, x):
        # This is the simple, original forward pass
        x = self.frontend(x)
        x = self.context(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)