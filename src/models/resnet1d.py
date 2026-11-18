"""
Advanced model: ResNet1D.

Key component written by us (course requirement): the Residual Block, y = F(x) + x.
Only primitive ops (Conv1d, BatchNorm1d, ReLU) come from PyTorch.
"""

import torch
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


# ResNet1D Architecture

class ResNet1D(nn.Module):

    def __init__(self, num_classes=18, in_channels=12):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 32)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


