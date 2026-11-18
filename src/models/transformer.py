"""Modern model: Transformer encoder for time series."""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=18,
        d_model=64,
        n_heads=4,
        n_layers=2,
        in_channels=12,
        dropout=0.1,
    ):
        super().__init__()

        # Project input features to transformer dimension
        self.input_projection = nn.Linear(in_channels, d_model)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 200, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: [B, 200, 12]
        x = self.input_projection(x)
        x = x + self.positional_encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output: [B, num_classes]
        return self.classifier(x)