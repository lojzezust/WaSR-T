"""
mobile_wasr_t.py

A version of wasr_t modified for inference on an embedded platform.

Identical to wasr_t except:
 - uses the smaller lrasp_mobilenet_v3 backbone
 - reduces the latent channel dimension of several elements in the decoder
 - trained for inference at 192x256 resolution

These changes are implemented as overrides through Python inheritance.
"""

from wasr_t.wasr_t import *
from wasr_t.layers import *

from torchvision.models.segmentation import lraspp_mobilenet_v3_large

model_urls = {
    'lraspp_mobilenet_v3_large' : 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
}

## Functional network definition

def wasr_temporal_lraspp_mobilenetv3(num_classes=3, pretrained=True, sequential=False, backbone_grad_steps=2, hist_len=5):

    # Pretrained LRASPP mobilenetv3 backbone
    backbone = lraspp_mobilenet_v3_large()

    # There are five non-convolutional backbone features in MobileNetV3.
    # 0: Feature 0 is (16, 192, 256)
    # 1: Feature 2 is (24, 96, 128)
    # 2: Feature 4 is (40, 48, 64)
    # 3: Feature 7 is (80, 24, 32).
    # 4: Feature 13 is (160, 12, 16).
    # 5: Feature 16 is (960, 12, 16).

    # skip connection locations determined by grad student descent
    skip1_pos = 2
    skip2_pos = 4
    aux_pos = 13
    out_pos = 16

    return_layers = {
        str(skip1_pos): "skip1",
        str(skip2_pos): "skip2",
        str(aux_pos): "aux",
        str(out_pos): "out"
    }
    return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone.backbone, return_layers=return_layers)

    decoder = MobileWaSRTDecoder(num_classes, hist_len=hist_len, sequential=sequential)

    model = WaSRT(backbone, decoder, backbone_grad_steps=backbone_grad_steps, sequential=sequential)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['lraspp_mobilenet_v3_large']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model

## The changes to the decoder follow

class MobileWaSRTDecoder(WaSRTDecoder):
    """A decoder module with reduced latent dimension, designed to work with wasrt_temporal_mobilenetv3."""

    def __init__(self, num_classes, hist_len=5, sequential=False):
        super().__init__(num_classes, hist_len, sequential)

        # Temporal Context Module
        self.tcm = MobileTemporalContextModule(960, hist_len=hist_len, sequential=sequential)

        self.arm1 = L.AttentionRefinementModule(240)
        self.arm2 = nn.Sequential(
            L.AttentionRefinementModule(40, last_arm=True),
            nn.Conv2d(40, 240, 1, 2) # Equalize number of features with ARM1
        )

        self.ffm = MobileFeatureFusionModule(24, 240, 128)
        self.aspp = L.ASPPv2(128, [6, 12, 18, 24], num_classes)

class MobileTemporalContextModule(TemporalContextModule):
    """A temporal context module with reduced latent dimension, designed to work with wasrt_temporal_mobilenetv3."""
    def __init__(self, in_features, hist_len=5, sequential=False):
        super().__init__(in_features, hist_len, sequential)

        div = 8 # arbitrary factor to reduce latent dimension, resulting in quadratic reduction of 3D conv complexity
        self.conv_in = nn.Conv2d(in_features, in_features//div, 1)
        self.conv_agg = nn.Conv3d(in_features//div, in_features//div, (hist_len+1, 3, 3), padding=(0,1,1))

class MobileFeatureFusionModule(FeatureFusionModule):
    """A feature fusion module with greater upsampling to account for reduced resolution in intermediate layers."""
    def __init__(self, bg_channels, sm_channels, num_features):
        super().__init__(bg_channels, sm_channels, num_features)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=4)