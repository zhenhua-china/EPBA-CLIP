import torch
import torch.nn as nn


class AdaptiveNoiseGenerator(nn.Module):
    """
    Generate adaptive noise sigma for M2IB bottleneck.

    Supported modes:
    1. fixed:      use a fixed scalar sigma
    2. learnable:  use one learnable sigma for the whole layer
    3. patchwise:  use one learnable sigma for each patch/token
    """

    def __init__(
        self,
        seq_len,
        noise_mode="fixed",
        sigma_fixed=1.0,
        sigma_min=0.05,
        sigma_max=1.0,
        device=None,
    ):
        super().__init__()

        assert noise_mode in ["fixed", "learnable", "patchwise"]

        self.seq_len = seq_len
        self.noise_mode = noise_mode
        self.sigma_fixed = sigma_fixed
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device

        if self.noise_mode == "learnable":
            
            self.sigma_param = nn.Parameter(
                torch.zeros(1, device=self.device)
            )

        elif self.noise_mode == "patchwise":
            
            self.sigma_param = nn.Parameter(
                torch.zeros(1, seq_len, 1, device=self.device)
            )

    def forward(self, x):
        """
        Args:
            x: feature map from a selected transformer layer.
               shape: [B, seq_len, hidden_dim]

        Returns:
            sigma: adaptive noise strength.
                   shape: [B, seq_len, 1]
        """
        B, N, D = x.shape

        if self.noise_mode == "fixed":
            sigma = torch.ones(B, N, 1, device=x.device) * self.sigma_fixed
            return sigma

        if self.noise_mode == "learnable":
            sigma_scalar = self.sigma_min + (
                self.sigma_max - self.sigma_min
            ) * torch.sigmoid(self.sigma_param)

            sigma = torch.ones(B, N, 1, device=x.device) * sigma_scalar
            return sigma

        if self.noise_mode == "patchwise":
            sigma = self.sigma_min + (
                self.sigma_max - self.sigma_min
            ) * torch.sigmoid(self.sigma_param)

            sigma = sigma.expand(B, -1, -1)
            return sigma
