"""
M — MDN-RNN (Mixture Density Network + RNN)
============================================
Predicts the DISTRIBUTION of the next latent z_{t+1},
not just a single point — it outputs params for a mixture of Gaussians.

Why a mixture? The world is stochastic. A single Gaussian can't
capture "the ball might go left OR right."

Architecture:
  Input:  [z_t, a_t]  →  LSTM  →  MDN head  →  (pi, mu, sigma)
  
The RNN hidden state h_t captures temporal context that z_t alone
can't encode (velocity, object permanence, enemy intent).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, a_dim=3, h_dim=256, n_mixtures=5):
        super().__init__()
        self.h_dim       = h_dim
        self.z_dim       = z_dim
        self.n_mixtures  = n_mixtures

        # RNN core
        self.rnn = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)

        # MDN head: outputs K*(2D+1) values per timestep
        #   K weights  +  K*D means  +  K*D sigmas
        self.mdn = nn.Linear(h_dim, n_mixtures * (2 * z_dim + 1))

    def forward(self, z_seq, a_seq, hidden=None):
        """
        z_seq: [batch, seq_len, z_dim]
        a_seq: [batch, seq_len, a_dim]
        Returns mixture params for each timestep.
        """
        x = torch.cat([z_seq, a_seq], dim=-1)   # [B, T, z+a]
        rnn_out, hidden = self.rnn(x, hidden)    # [B, T, h_dim]

        raw = self.mdn(rnn_out)                  # [B, T, K*(2D+1)]
        pi, mu, sigma = self._split_mdn(raw)
        return pi, mu, sigma, hidden

    def _split_mdn(self, raw):
        """
        Split raw MDN output into mixture weights, means, and stds.
        For K mixtures and D latent dims:
          - pi:    [B, T, K]           — mixture weights
          - mu:    [B, T, K, D]        — means
          - sigma: [B, T, K, D]        — standard deviations
        """
        K, D = self.n_mixtures, self.z_dim

        pi    = raw[..., :K]                                  # logits → softmax later
        mu    = raw[..., K : K + K * D].view(*raw.shape[:-1], K, D)
        sigma = raw[..., K + K * D :  ].view(*raw.shape[:-1], K, D)

        pi    = F.softmax(pi, dim=-1)
        sigma = torch.exp(sigma).clamp(min=1e-4)   # must be positive
        return pi, mu, sigma

    def mdn_loss(self, pi, mu, sigma, z_target):
        """
        Negative log-likelihood under the mixture model.
        z_target: [B, T, D]
        """
        z_target = z_target.unsqueeze(-2)          # [B, T, 1, D]

        # Gaussian log-prob for each mixture component
        log_prob = -0.5 * (((z_target - mu) / sigma) ** 2
                           + 2 * sigma.log()
                           + torch.log(torch.tensor(2 * 3.14159)))
        log_prob = log_prob.sum(-1)                # sum over D → [B, T, K]

        # Weight by mixture probabilities and sum
        log_mix  = torch.log(pi + 1e-8) + log_prob
        loss     = -torch.logsumexp(log_mix, dim=-1)  # [B, T]
        return loss.mean()

    def sample_next_z(self, pi, mu, sigma):
        """
        Sample z_{t+1} from the predicted mixture at a single timestep.
        pi, mu, sigma: last-timestep values
        """
        k = torch.multinomial(pi, 1).squeeze(-1)   # pick a mixture
        # Gather selected component's mu and sigma
        mu_k    = mu[torch.arange(len(k)), k]
        sigma_k = sigma[torch.arange(len(k)), k]
        return mu_k + sigma_k * torch.randn_like(mu_k)


# --- Quick demo ---
if __name__ == "__main__":
    B, T  = 4, 20   # batch=4, sequence length=20
    model = MDNRNN(z_dim=32, a_dim=3, h_dim=256, n_mixtures=5)

    z_seq = torch.randn(B, T, 32)
    a_seq = torch.randn(B, T, 3)

    pi, mu, sigma, hidden = model(z_seq, a_seq)

    print(f"pi shape:    {pi.shape}")      # [4, 20, 5]
    print(f"mu shape:    {mu.shape}")      # [4, 20, 5, 32]
    print(f"sigma shape: {sigma.shape}")   # [4, 20, 5, 32]

    # Train loss: predict z_{t+1} from z_t
    z_target = z_seq[:, 1:, :]    # shift by 1
    pi_, mu_, sigma_, _ = model(z_seq[:, :-1, :], a_seq[:, :-1, :])
    loss = model.mdn_loss(pi_, mu_, sigma_, z_target)
    print(f"MDN loss:    {loss.item():.4f}")
