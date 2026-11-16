"""Advanced model: ResNet1D.

Key component written by us (course requirement): the Residual Block, y = F(x) + x.
Only primitive ops (Conv1d, BatchNorm1d, ReLU) come from PyTorch.
"""
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        # projection shortcut when the shape changes
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))   # F(x)
        out = out + self.shortcut(x)      # F(x) + x  (skip connection)
        return self.relu(out)


# TODO: ResNet1D that stacks ResidualBlock1D + global pooling + classifier
