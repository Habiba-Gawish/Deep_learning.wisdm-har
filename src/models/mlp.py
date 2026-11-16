"""Baseline: Small MLP (2400 -> 128 -> 64 -> 18).

Input : [B, 200, 12]  (flattened to 2400 inside the model)
Output: logits [B, num_classes]  (softmax is applied inside CrossEntropyLoss)
"""
import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self, in_features=2400, num_classes=18, hidden=(128, 64)):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)