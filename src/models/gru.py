"""Classic model: bidirectional GRU (recurrent network).

Input : [B, 200, 12]  (batch_first, no permute needed)
Output: logits [B, num_classes]  (softmax is applied inside CrossEntropyLoss)
"""
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, num_classes=18, in_channels=12, hidden_size=64,
                 num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout between GRU layers
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)   # [B, 200, out_dim]: hidden state at every time step
        out = out.mean(dim=1)  # average over time -> [B, out_dim]
        return self.classifier(out)
