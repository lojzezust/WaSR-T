
from collections import OrderedDict
import contextlib

import torch
from torch import nn
from torchvision.models.resnet import resnet101
from torch.hub import load_state_dict_from_url

from wasr_t.utils import IntermediateLayerGetter
import wasr_t.layers as L

model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}

def wasr_temporal_resnet101(num_classes=3, pretrained=True, sequential=False, backbone_grad_steps=2, hist_len=5):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    decoder = WaSRTDecoder(num_classes, hist_len=hist_len, sequential=sequential)

    model = WaSRT(backbone, decoder, backbone_grad_steps=backbone_grad_steps, sequential=sequential)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model

class WaSRT(nn.Module):
    """WaSR-T model"""
    def __init__(self, backbone, decoder, backbone_grad_steps=2, sequential=False):
        super(WaSRT, self).__init__()

        self.backbone = backbone
        self.decoder = decoder
        self.backbone_grad_steps = backbone_grad_steps

        self._is_sequential = sequential

    def forward(self, x):
        if self._is_sequential:
            return self.forward_sequential(x)
        else:
            return self.forward_unrolled(x)

    def forward_sequential(self, x):
        features = self.backbone(x['image'])

        x = self.decoder(features)

        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

    def forward_unrolled(self, x):
        features = self.backbone(x['image'])

        extract_feats = ['out','skip1','skip2']
        feats_hist = {f:[] for f in extract_feats}
        hist_len = x['hist_images'].shape[1]
        for i in range(hist_len):
            # Compute gradients only in last backbone_grad_steps - 1 steps
            use_grad = i >= hist_len - self.backbone_grad_steps + 1
            ctx = contextlib.nullcontext() if use_grad else torch.no_grad()
            with ctx:
                feats = self.backbone(x['hist_images'][:,i])
                for f in extract_feats:
                   feats_hist[f].append(feats[f])

        # Stack tensors
        for f in extract_feats:
            feats_hist[f] = torch.stack(feats_hist[f], 1)

        x = self.decoder(features, feats_hist)

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output

    def sequential(self):
        """Switch network to sequential mode."""

        self._is_sequential = True
        self.decoder.sequential()

        return self

    def unrolled(self):
        """Switch network to unrolled mode."""

        self._is_sequential = False
        self.decoder.unrolled()

        return self

    def clear_state(self):
        """Clears state of the network. Used to reset model between sequences in sequential mode."""

        self.decoder.clear_state()


class WaSRTDecoder(nn.Module):
    def __init__(self, num_classes, hist_len=5, sequential=False):
        super(WaSRTDecoder, self).__init__()

        self.arm1 = L.AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            L.AttentionRefinementModule(512, last_arm=True),
            nn.Conv2d(512, 2048, 1) # Equalize number of features with ARM1
        )

        # Temporal Context Module
        self.tcm = L.TemporalContextModule(2048, hist_len=hist_len, sequential=sequential)

        self.ffm = L.FeatureFusionModule(256, 2048, 1024)
        self.aspp = L.ASPPv2(1024, [6, 12, 18, 24], num_classes)

    def forward(self, x, x_hist=None):
        if x_hist is None: x_hist={'skip1':None, 'skip2': None, 'out': None}
        feats_out = self.tcm(x['out'], x_hist['out'])

        arm1 = self.arm1(feats_out)
        arm2 = self.arm2(x['skip2'])
        arm_combined = arm1 + arm2

        x = self.ffm(x['skip1'], arm_combined)

        output = self.aspp(x)

        return output

    def clear_state(self):
        self.tcm.clear_state()

    def sequential(self):
        """Switch to sequential mode."""

        self.tcm.sequential()

        return self

    def unrolled(self):
        """Switch to unrolled mode."""

        self.tcm.unrolled()

        return self
