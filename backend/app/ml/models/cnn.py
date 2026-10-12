"""
CNN classifier: mel-spectrogram -> propeller anomaly severity class.

Input:  (batch, 1, n_mels, n_frames)  e.g. (B, 1, 64, 126) for the default config
Output: (batch, num_classes) logits

Architecture: 4 conv blocks (conv -> BN -> ReLU -> maxpool) + global average
pool + a small classifier head. Deliberately compact (~106K params, see models/registry.json) because
we are training on ~13k segments on a single CPU core, not a GPU cluster —
a large architecture would overfit and be impractically slow to train here.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class AeroSenseCNN(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),   # -> /2
            ConvBlock(16, 32),            # -> /4
            ConvBlock(32, 64),            # -> /8
            ConvBlock(64, 128),           # -> /16
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled 128-d embedding before the classifier head
        (used for embedding-space unknown-anomaly distance scoring)."""
        x = self.features(x)
        x = self.gap(x)
        return torch.flatten(x, 1)
