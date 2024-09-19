import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class DoubleConv(nn.Module):
    """
    Since at each step of the encoder, we perform 3x3 conv
    twice, we can create a class for this and do just that.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            # Same convolution. Height and width remain same.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),   # bias false because using batchNorm.
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),   # bias false because using batchNorm.
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        
        return down, p
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        """
        x1: feature maps from previous layers.
        x2: feature maps carried over using skip connections.
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)  # Adding the skip connection.
        return self.conv(x)