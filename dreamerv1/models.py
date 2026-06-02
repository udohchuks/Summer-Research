# Deterministic State (h_t): Maintained by a standard GRU/LSTM. It keeps track of long-term history.
# Stochastic State (z_t): A probability distribution (Gaussian/Normal) that captures the instantaneous uncertainty of the current moment.
# The RSSM has two modes of operation:
# Posterior (Representation) Mode: Used during training when we have the actual image x_t from the real environment.
# It combines the past state and the current image to guess the true latent state.
# Prior (Transition) Mode: Used during "imagination". It guesses what happens next without looking at a real image,
# relying only on the previous state and the action taken.
# models.py
import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

class RSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stochastic_dim = config.world_model.stochastic_dim
        self.deterministic_dim = config.world_model.deterministic_dim
        self.hidden_dim = config.world_model.hidden_dim
        self.embed_dim = config.world_model.embed_dim
        self.action_dim = config.environment.action_dim
        
        # 1. Recurrent core (GRU)
        self.gru_cell = nn.GRUCell(self.stochastic_dim + self.action_dim, self.deterministic_dim)
        
        # 2. Prior Network: predicting z_t from h_t
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.deterministic_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim)
        )
        
        # 3. Posterior Network: predicting z_t from h_t AND image embedding e_t
        self.posterior_mlp = nn.Sequential(
            nn.Linear(self.deterministic_dim + self.embed_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.stochastic_dim)
        )

    def formalize_dist(self, mean_std_tensor):
        mean, raw_std = torch.chunk(mean_std_tensor, 2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + 0.1 
        return Normal(mean, std)

    def prior_step(self, prev_stochastic, action):
        """Imagination step: Predicts next state purely from history and action."""
        gru_input = torch.cat([prev_stochastic, action], dim=-1)
        hidden_state = self.gru_cell(gru_input) 
        
        prior_params = self.prior_mlp(hidden_state)
        prior_dist = self.formalize_dist(prior_params)
        stochastic_sample = prior_dist.rsample()
        
        return hidden_state, stochastic_sample, prior_dist

    def posterior_step(self, hidden_state, image_embedding):
        """Observation step: Updates state belief based on real environment image."""
        post_input = torch.cat([hidden_state, image_embedding], dim=-1)
        posterior_params = self.posterior_mlp(post_input)
        posterior_dist = self.formalize_dist(posterior_params)
        stochastic_sample = posterior_dist.rsample()
        
        return stochastic_sample, posterior_dist


class ConvEncoder(nn.Module):
    """Encodes a [3, 64, 64] image into a flat latent embedding vector."""
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.world_model.embed_dim
        
        self.net = nn.Sequential(
            nn.Conv2d(config.environment.image_shape[0], 32, kernel_size=4, stride=2), # -> [32, 31, 31]
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                                # -> [64, 14, 14]
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),                               # -> [128, 6, 6]
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),                              # -> [256, 2, 2]
            nn.ELU(),
            nn.Flatten()                                                               # -> 256 * 2 * 2 = 1024
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    """Decodes combined latent states (h_t + z_t) back into a reconstructed image."""
    def __init__(self, config):
        super().__init__()
        in_dim = config.world_model.deterministic_dim + config.world_model.stochastic_dim
        
        self.fc = nn.Linear(in_dim, 1024)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), # -> [128, 5, 5]
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),  # -> [64, 13, 13]
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),   # -> [32, 30, 30]
            nn.ELU(),
            nn.ConvTranspose2d(32, config.environment.image_shape[0], kernel_size=6, stride=2), # -> [3, 64, 64]
        )

    def forward(self, latent_state):
        x = self.fc(latent_state)
        x = x.view(-1, 1024, 1, 1) # Reshape to feed into Transpose Convolutions
        return self.net(x)



class RewardPredictor(nn.Module):
    """Predicts a scalar reward from the combined latent state (h_t + z_t)."""
    def __init__(self, config):
        super().__init__()
        in_dim = config.world_model.deterministic_dim + config.world_model.stochastic_dim
        hidden_dim = config.world_model.hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar value
        )

    def forward(self, latent_state):
        return self.net(latent_state)
    


class ActionActor(nn.Module):
    """The Policy. Predicts actions given the latent state (h_t + z_t) 
       and correctly squashes them using a TanhNormal distribution."""
    def __init__(self, config):
        super().__init__()
        in_dim = config.world_model.deterministic_dim + config.world_model.stochastic_dim
        hidden_dim = config.world_model.hidden_dim
        self.action_dim = config.environment.action_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * self.action_dim)
        )

    def forward(self, latent_state):
        stats = self.net(latent_state)
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        
        # Enforce stable mean and standard deviation limits
        mean = torch.clamp(mean, min=-5.0, max=5.0) 
        std = torch.nn.functional.softplus(raw_std) + 0.1
        
        # 1. Create the base normal distribution
        base_dist = Normal(mean, std)
        
        # 2. Apply the Tanh transform. 
        # cache_size=1 is an internal PyTorch optimization for tracking gradients during rsample()
        tanh_transform = TanhTransform(cache_size=1)
        
        # 3. Combine them into a TanhNormal distribution
        squashed_dist = TransformedDistribution(base_dist, [tanh_transform])
        
        # This samples using the reparameterization trick AND applies the tanh squashing!
        action = squashed_dist.rsample() 
        
        return action, squashed_dist


class ValueCritic(nn.Module):
    """The Value Function. Predicts expected long-term value from latent state."""
    def __init__(self, config):
        super().__init__()
        in_dim = config.world_model.deterministic_dim + config.world_model.stochastic_dim
        hidden_dim = config.world_model.hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent_state):
        return self.net(latent_state)