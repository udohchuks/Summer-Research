"""
C — Controller
==============
The simplest possible piece of the World Models architecture.
A single linear layer that maps [z_t, h_t] → action.

Why so tiny?
  V and M do all the heavy lifting — understanding the world.
  C just needs to learn *which action to take* given a rich state.
  Because it's tiny, it can be optimized with evolution strategies
  (e.g., CMA-ES) instead of gradient descent — no backprop needed.

Input:  z_t  (32-dim VAE latent)  +  h_t  (256-dim RNN hidden state)
Output: action (continuous or discrete depending on the environment)
"""

import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, z_dim=32, h_dim=256, a_dim=3):
        super().__init__()
        # The entire controller is one linear layer
        self.fc = nn.Linear(z_dim + h_dim, a_dim)

    def forward(self, z, h):
        """
        z: [batch, z_dim]   — VAE latent for current frame
        h: [batch, h_dim]   — RNN hidden state (temporal memory)
        Returns action: [batch, a_dim]
        """
        x      = torch.cat([z, h], dim=-1)  # concatenate z and h
        action = torch.tanh(self.fc(x))     # tanh to bound actions in [-1, 1]
        return action

    def get_flat_params(self):
        """Flatten all params into a 1D vector (for CMA-ES)."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params):
        """Load a 1D vector of params back into the controller (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view(p.shape))
            idx += n


# --- Quick demo ---
if __name__ == "__main__":
    ctrl = Controller(z_dim=32, h_dim=256, a_dim=3)

    z      = torch.randn(1, 32)    # single frame latent
    h      = torch.randn(1, 256)   # RNN hidden state
    action = ctrl(z, h)

    n_params = sum(p.numel() for p in ctrl.parameters())
    print(f"Action:          {action.shape}  →  {action.squeeze().tolist()}")
    print(f"Total params:    {n_params}")          # (32+256)*3 + 3 = 867
    print(f"Flat param vec:  {ctrl.get_flat_params().shape}")
