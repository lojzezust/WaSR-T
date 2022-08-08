import torch
from torch import nn

def time_independent_forward(module, x):
    # Squash samples and timesteps into a single axis
    B,T,*S = x.shape
    x_reshape = x.contiguous().view(B*T, *S)  # (batch * timesteps, input_size)

    y = module(x_reshape)

    # Reshape back into batch and timesteps
    _,*S = y.shape
    y = y.contiguous().view(B, T, *S)  # (batch, timesteps, input_size)

    return y

class TemporalContextModule(nn.Module):
    """Stores a running memory of previoius features."""
    def __init__(self, in_features, hist_len=5, sequential=False):
        super(TemporalContextModule, self).__init__()

        self.conv_in = nn.Conv2d(in_features, in_features//2, 1)
        self.conv_agg = nn.Conv3d(in_features//2, in_features//2, (hist_len+1, 3, 3), padding=(0,1,1))

        self.hist_len = hist_len
        self._is_sequential = sequential
        self._sequential_mem = None


    def forward(self, feat, feat_mem=None):
        if self._is_sequential:
            return self.forward_sequential(feat)
        else:
            return self.forward_unrolled(feat, feat_mem)

    def clear_state(self):
        """Clears feature memory. Should be called before inference on a new sequence."""

        self._sequential_mem = None

    def sequential(self):
        """Switch to sequential mode."""

        self._is_sequential = True
        self._sequential_mem = None
        return self

    def unrolled(self):
        """Switch to unrolled mode."""

        self._is_sequential = False
        self._sequential_mem = None
        return self

    def _aggregate(self, hist_volume):
        # Avg pool aggregation
        agg = self.conv_agg(hist_volume.permute(0,2,1,3,4)).squeeze(2)

        out = torch.cat([agg, hist_volume[:,-1]], 1)
        return out

    def forward_sequential(self, feat):
        assert feat.size(0) == 1, "Batch size should be 1 for sequential inference."
        feat_in = self.conv_in(feat)
        if self._sequential_mem is None:
            self._sequential_mem = feat_in.unsqueeze(1).repeat(1,self.hist_len,1,1,1)

        hist_volume = torch.cat([self._sequential_mem, feat_in.unsqueeze(1)], dim=1)

        # Discard oldest frame from memory
        self._sequential_mem = hist_volume[:, 1:]

        return self._aggregate(hist_volume)

    def forward_unrolled(self, feat, feat_mem):
        hist_volume = torch.cat([feat_mem, feat.unsqueeze(1)], dim=1)
        hist_volume = time_independent_forward(self.conv_in, hist_volume)

        return self._aggregate(hist_volume)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, last_arm=False):
        super(AttentionRefinementModule, self).__init__()

        self.last_arm = last_arm

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x

        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        weights = self.sigmoid(x)

        out = weights * input

        if self.last_arm:
            weights = self.global_pool(out)
            out = weights * out

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, bg_channels, sm_channels, num_features):
        super(FeatureFusionModule, self).__init__()

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_features, num_features, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_big, x_small):
        if x_big.size(2) > x_small.size(2):
            x_small = self.upsampling(x_small)

        x = torch.cat((x_big, x_small), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_out = self.relu(x)

        x = self.global_pool(conv1_out)
        x = self.conv2(x)
        x = self.conv3(x)
        weights = self.sigmoid(x)

        mul = weights * conv1_out
        out = conv1_out + mul

        return out

class ASPPv2Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bias=False, bn=False, relu=False):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=bias))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))

        if relu:
            modules.append(nn.ReLU())

        super(ASPPv2Conv, self).__init__(*modules)

class ASPPv2(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, relu=False, biased=True):
        super(ASPPv2, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPv2Conv(in_channels, out_channels, rate, bias=biased, relu=relu))

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Sum convolution results
        res = torch.stack(res).sum(0)
        return res

