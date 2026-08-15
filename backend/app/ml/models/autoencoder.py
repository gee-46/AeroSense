"""
Convolutional autoencoder for anomaly detection.

Trained ONLY on `normal` mel-spectrogram segments. At inference time, the
reconstruction error (MSE between input and reconstruction) is high for
segments the model has never seen the acoustic pattern of — i.e. faulty
propellers — because the encoder/decoder only learned to compress normal
propeller acoustics well.

The anomaly threshold is NOT hardcoded here; it is fit in
evaluation/evaluate.py from the distribution of validation-set
reconstruction errors and stored in models/registry.json.
"""
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Conv2d(64, latent_dim, 1)
        self.expand = nn.Conv2d(latent_dim, 64, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor):
        target_shape = x.shape[-2:]
        z = self.encoder(x)
        z = self.bottleneck(z)
        y = self.expand(z)
        recon = self.decoder(y)
        # Guard against off-by-one size mismatches from odd input dims.
        if recon.shape[-2:] != target_shape:
            recon = nn.functional.interpolate(recon, size=target_shape, mode="bilinear", align_corners=False)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error. Shape: (batch,)"""
        recon = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=[1, 2, 3])
