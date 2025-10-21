"""
3D U-Net implementation for volumetric segmentation of teeth in µCT scans.
Segments three classes: enamel (Zahnschmelz), dentin, and pulpa.
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is NCDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2,
                                     diffH // 2, diffH - diffH // 2,
                                     diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.
    
    Args:
        n_channels: Number of input channels (typically 1 for grayscale µCT)
        n_classes: Number of output classes (4: background, enamel, dentin, pulpa)
        base_channels: Base number of channels (default: 64)
        trilinear: Use trilinear upsampling instead of transposed convolutions
    """
    
    def __init__(self, n_channels=1, n_classes=4, base_channels=64, trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv3D(n_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, trilinear)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, trilinear)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, trilinear)
        self.up4 = Up3D(base_channels * 2, base_channels, trilinear)
        self.outc = OutConv3D(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)


if __name__ == "__main__":
    # Test the model
    model = UNet3D(n_channels=1, n_classes=4, base_channels=32)
    x = torch.randn(1, 1, 64, 64, 64)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
