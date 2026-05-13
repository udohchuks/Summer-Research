"""
V — Variational Autoencoder
===========================
Compresses raw 64x64x3 frames into a 32-dim latent vector z.
The latent space is structured (via KL loss) so the MDN-RNN
can meaningfully predict future z values.

Loss = Reconstruction (MSE) + KL Divergence
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: image → (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2),  # 64→31
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),            # 31→14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),           # 14→6
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),          # 6→2
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu     = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder: z → reconstructed image
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 6, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode image to (mu, log_var)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
          z = mu + sigma * epsilon,  epsilon ~ N(0,1)
        This lets gradients flow through the random sampling step.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent z back to image space."""
        h = self.fc_decode(z).view(-1, 256, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss(self, x, recon, mu, log_var):
        """
        VAE loss = Reconstruction + KL Divergence
          - MSE ensures decoder rebuilds the image accurately
          - KL keeps z well-behaved so the RNN can predict it
        """
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss


# --- Quick demo ---
if __name__ == "__main__":
    vae = VAE(img_channels=3, latent_dim=32)
    x   = torch.randn(4, 3, 64, 64)          # batch of 4 frames

    recon, mu, log_var = vae(x)
    loss = vae.loss(x, recon, mu, log_var)

    print(f"Input:         {x.shape}")
    print(f"Latent z (mu): {mu.shape}")       # [4, 32]
    print(f"Reconstructed: {recon.shape}")    # [4, 3, 64, 64]
    print(f"Loss:          {loss.item():.2f}")
