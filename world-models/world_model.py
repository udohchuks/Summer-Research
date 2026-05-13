"""
World Model — Full Pipeline (V + M + C)
========================================
Combines the three components:
  V (VAE)     — compresses observations → latent z
  M (MDN-RNN) — predicts next z, maintains hidden state h
  C (Controller) — maps [z, h] → action

Data flow at each timestep t:
  obs_t  →  V.encode  →  z_t
  [z_t, h_t]  →  C  →  action_t
  [z_t, action_t]  →  M  →  h_{t+1}, predicted z_{t+1}

Training is done in phases (not end-to-end):
  Phase 1: Train V on real observations (random rollouts)
  Phase 2: Encode data with frozen V; train M on latent sequences
  Phase 3: Freeze V+M; optimize C with CMA-ES in dream env
"""

import torch
import torch.nn as nn

from vae      import VAE
from mdn_rnn  import MDNRNN
from controller import Controller


class WorldModel(nn.Module):
    def __init__(self, z_dim=32, h_dim=256, a_dim=3, n_mixtures=5):
        super().__init__()
        self.V = VAE(img_channels=3, latent_dim=z_dim)
        self.M = MDNRNN(z_dim=z_dim, a_dim=a_dim, h_dim=h_dim, n_mixtures=n_mixtures)
        self.C = Controller(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

    # ------------------------------------------------------------------
    # Real environment loop
    # ------------------------------------------------------------------
    def act(self, obs, hidden=None):
        """
        Given a raw observation and optional RNN hidden state,
        return an action and the updated hidden state.

        obs:    [1, C, H, W]  — single raw frame
        hidden: LSTM (h, c) tuple or None
        """
        with torch.no_grad():
            mu, _ = self.V.encode(obs)             # z_t = mu (no sampling at test time)
            z = mu

            h = hidden[0].squeeze(0) if hidden else torch.zeros(1, self.h_dim)
            action = self.C(z, h)                  # [1, a_dim]

            # Step M to update hidden state
            z_seq = z.unsqueeze(1)                 # [1, 1, z_dim]
            a_seq = action.unsqueeze(1)            # [1, 1, a_dim]
            _, _, _, hidden = self.M(z_seq, a_seq, hidden)

        return action, hidden

    # ------------------------------------------------------------------
    # Dream (hallucination) loop
    # ------------------------------------------------------------------
    def dream(self, z0, hidden=None, n_steps=100, temperature=1.0):
        """
        Run the agent purely inside its imagination.
        No real environment needed — M predicts the next z.

        z0:     [1, z_dim]  — initial latent
        Returns trajectory of (z, action) pairs.
        """
        trajectory = []
        z = z0

        for _ in range(n_steps):
            h = hidden[0].squeeze(0) if hidden else torch.zeros(1, self.h_dim)
            action = self.C(z, h)

            # Sample next z from M's predicted distribution
            z_seq = z.unsqueeze(1)
            a_seq = action.unsqueeze(1)
            pi, mu, sigma, hidden = self.M(z_seq, a_seq, hidden)

            # Sample from the mixture (with temperature)
            sigma_scaled = sigma * temperature
            z = self.M.sample_next_z(
                pi[:, 0, :],
                mu[:, 0, :, :],
                sigma_scaled[:, 0, :, :]
            )
            trajectory.append((z.detach(), action.detach()))

        return trajectory

    # ------------------------------------------------------------------
    # Phase 1: Train V
    # ------------------------------------------------------------------
    def train_vae(self, obs_batch, optimizer):
        """One gradient step on the VAE using real observations."""
        recon, mu, log_var = self.V(obs_batch)
        loss = self.V.loss(obs_batch, recon, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Phase 2: Train M (V frozen)
    # ------------------------------------------------------------------
    def train_mdn_rnn(self, z_seq, a_seq, optimizer):
        """
        One gradient step on the MDN-RNN.
        V must already be trained; z_seq is pre-encoded.
        Predicts z_{t+1} from z_t, a_t.
        """
        pi, mu, sigma, _ = self.M(z_seq[:, :-1, :], a_seq[:, :-1, :])
        loss = self.M.mdn_loss(pi, mu, sigma, z_seq[:, 1:, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


# ------------------------------------------------------------------
# Quick demo — forward pass + dream
# ------------------------------------------------------------------
if __name__ == "__main__":
    wm = WorldModel(z_dim=32, h_dim=256, a_dim=3, n_mixtures=5)

    # --- Real environment step ---
    obs    = torch.randn(1, 3, 64, 64)
    action, hidden = wm.act(obs)
    print(f"[Real env] action: {action.shape}, hidden h: {hidden[0].shape}")

    # --- Phase 1 quick check ---
    obs_batch = torch.randn(8, 3, 64, 64)
    vae_opt   = torch.optim.Adam(wm.V.parameters(), lr=1e-4)
    v_loss    = wm.train_vae(obs_batch, vae_opt)
    print(f"[Phase 1] VAE loss: {v_loss:.2f}")

    # --- Phase 2 quick check ---
    with torch.no_grad():
        mu, _ = wm.V.encode(obs_batch)
    z_seq  = mu.unsqueeze(1).expand(-1, 20, -1)  # fake seq for demo
    a_seq  = torch.randn(8, 20, 3)
    mdn_opt = torch.optim.Adam(wm.M.parameters(), lr=1e-4)
    m_loss  = wm.train_mdn_rnn(z_seq, a_seq, mdn_opt)
    print(f"[Phase 2] MDN-RNN loss: {m_loss:.4f}")

    # --- Dream rollout ---
    z0         = torch.randn(1, 32)
    trajectory = wm.dream(z0, n_steps=10)
    print(f"[Dream]   {len(trajectory)} steps, z shape: {trajectory[0][0].shape}")
