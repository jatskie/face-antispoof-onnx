"""MiniFASNet V2 SE architecture with Fourier Transform auxiliary head.

Ported from minivision-ai/Silent-Face-Anti-Spoofing. Attribute names are load-bearing:
they are the state-dict keys of the released checkpoints.
"""

import torch
from torch import nn

# Channel widths of the 1.8M-parameter MiniFASNetV2. A depthwise block starting at
# index i maps CHANNELS[i] -> [i+1] (1x1 expand) -> [i+2] (3x3 depthwise) -> [i+3] (1x1 project).
# fmt: off
CHANNELS = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64,
    231, 231, 128, 231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128,
    26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
]
# fmt: on


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> optional PReLU."""

    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c) if activation else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.prelu(x) if self.prelu is not None else x


class SEModule(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.relu(self.bn1(self.fc1(self.avg_pool(x))))
        return x * self.sigmoid(self.bn2(self.fc2(w)))


class DepthWise(nn.Module):
    """Inverted-bottleneck block: 1x1 expand, 3x3 depthwise, 1x1 linear projection."""

    def __init__(self, channels, stride=1, residual=False, se=False):
        super().__init__()
        c_in, c_expand, c_dw, c_out = channels
        self.conv = ConvBlock(c_in, c_expand)
        self.conv_dw = ConvBlock(
            c_expand, c_dw, kernel=3, stride=stride, padding=1, groups=c_expand
        )
        self.project = ConvBlock(c_dw, c_out, activation=False)
        self.se_module = SEModule(c_out, reduction=4) if se else None
        self.residual = residual

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        if self.se_module is not None:
            out = self.se_module(out)
        return x + out if self.residual else out


class Residual(nn.Module):
    """`num_blocks` residual depthwise blocks starting at CHANNELS[start]; the last has SE."""

    def __init__(self, start, num_blocks):
        super().__init__()
        self.model = nn.Sequential(
            *(
                DepthWise(
                    CHANNELS[start + 3 * n : start + 3 * n + 4],
                    residual=True,
                    se=n == num_blocks - 1,
                )
                for n in range(num_blocks)
            )
        )

    def forward(self, x):
        return self.model(x)


class MiniFASNetV2SE(nn.Module):
    def __init__(
        self,
        embedding_size=128,
        conv6_kernel=(7, 7),
        dropout_prob=0.75,
        num_classes=2,
        num_channels=3,
    ):
        super().__init__()
        c = CHANNELS
        self.conv1 = ConvBlock(num_channels, c[0], kernel=3, stride=2, padding=1)
        self.conv2_dw = ConvBlock(c[0], c[1], kernel=3, padding=1, groups=c[1])
        self.conv_23 = DepthWise(c[1:5], stride=2)
        self.conv_3 = Residual(start=4, num_blocks=4)
        self.conv_34 = DepthWise(c[16:20], stride=2)
        self.conv_4 = Residual(start=19, num_blocks=6)
        self.conv_45 = DepthWise(c[37:41], stride=2)
        self.conv_5 = Residual(start=40, num_blocks=2)
        self.conv_6_sep = ConvBlock(c[46], c[47])
        self.conv_6_dw = ConvBlock(
            c[47], c[48], kernel=conv6_kernel, groups=c[48], activation=False
        )
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.logits = nn.Linear(embedding_size, num_classes, bias=False)

    def features(self, x):
        """Backbone up to conv_4: the feature map the FT branch taps during training."""
        x = self.conv2_dw(self.conv1(x))
        return self.conv_4(self.conv_34(self.conv_3(self.conv_23(x))))

    def head(self, x):
        x = self.conv_6_dw(self.conv_6_sep(self.conv_5(self.conv_45(x))))
        x = self.bn(self.linear(x.flatten(1)))
        return self.logits(self.dropout(x))

    def forward(self, x):
        return self.head(self.features(x))


class FTGenerator(nn.Module):
    """Predicts the input's Fourier spectrum from backbone features (training only)."""

    def __init__(self, in_channels=128, out_channels=1):
        super().__init__()
        self.fourier_transform = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fourier_transform(x)


class MultiFTNet(nn.Module):
    """MiniFAS classifier; in train mode also returns the FT branch's spectrum prediction."""

    def __init__(
        self, num_channels=3, num_classes=2, embedding_size=128, conv6_kernel=(5, 5)
    ):
        super().__init__()
        self.model = MiniFASNetV2SE(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            num_channels=num_channels,
        )
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        features = self.model.features(x)
        logits = self.model.head(features)
        if self.training:
            return logits, self.FTGenerator(features)
        return logits


def load_weights(model, checkpoint, include_ft=True):
    """Load a checkpoint (path or dict) into an unwrapped MultiFTNet.

    Accepts every format this repo has produced: wrapped in "model_state_dict" or raw,
    with or without a DataParallel "module." prefix, and older ".prob"/"FTGenerator.ft"
    key names. Only the training-only FT branch may be missing; any other mismatch raises.
    """
    if not isinstance(checkpoint, dict):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        key = key.replace(".prob.", ".logits.")
        key = key.replace("FTGenerator.ft.", "FTGenerator.fourier_transform.")
        if include_ft or not key.startswith("FTGenerator."):
            cleaned[key] = value

    result = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("FTGenerator.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not match model: missing={missing}, "
            f"unexpected={result.unexpected_keys}"
        )
    return checkpoint
