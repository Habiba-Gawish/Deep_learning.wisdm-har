"""Classic model: 1D-CNN.

Input : [B, 200, 12]  (permuted to [B, 12, 200] inside forward)
Output: logits [B, num_classes]
"""
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, num_classes=18, in_channels=12, dropout=0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        # time length: 200 -> 100 -> 50 -> 25
        self.features = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over time
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)         # [B, 200, 12] -> [B, 12, 200]
        x = self.features(x)           # [B, 256, 25]
        x = self.pool(x).squeeze(-1)   # [B, 256]
        return self.classifier(x)      # [B, num_classes]