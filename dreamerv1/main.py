# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence

from utils import load_config
from models import RSSM, ConvEncoder, ConvDecoder, RewardPredictor
from buffer import SequenceReplayBuffer

def train_world_model_step():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Initialize Networks & Optimizer
    encoder = ConvEncoder(config).to(device)
    decoder = ConvDecoder(config).to(device)
    rssm = RSSM(config).to(device)
    reward_predictor = RewardPredictor(config).to(device)
    
    # Group all parameters together for a single optimizer
    model_params = (list(encoder.parameters()) + 
                    list(decoder.parameters()) + 
                    list(rssm.parameters()) + 
                    list(reward_predictor.parameters()))
    optimizer = optim.Adam(model_params, lr=3e-4)
    
    # 2. Populate Buffer with fake data for testing
    buffer = SequenceReplayBuffer(config)
    for _ in range(100):
        buffer.add(
            np_img := torch.randn(*config.environment.image_shape).numpy(),
            np_act := torch.randn(config.environment.action_dim).numpy(),
            np_rew := [1.0],
            np_term := [0.0]
        )
        
    # 3. Sample a sequential batch
    obs_seq, act_seq, rew_seq, term_seq = buffer.sample(device)
    
    batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
    
    # Initialize hidden states
    h_t = torch.zeros(batch_size, config.world_model.deterministic_dim, device=device)
    z_t = torch.zeros(batch_size, config.world_model.stochastic_dim, device=device)
    
    # Accumulators for losses over the sequence
    recon_loss = 0.0
    reward_loss = 0.0
    kl_loss = 0.0
    
    # 4. Unroll over the sequence and compute losses step-by-step
    for t in range(seq_len):
        img_t = obs_seq[:, t]
        act_t = act_seq[:, t]
        rew_t = rew_seq[:, t]
        
        # Step A: Prior step
        h_t, prior_z, prior_dist = rssm.prior_step(z_t, act_t)
        
        # Step B: Posterior step
        embed_t = encoder(img_t)
        z_t, post_dist = rssm.posterior_step(h_t, embed_t)
        
        # Combined latent state representation
        latent_t = torch.cat([h_t, z_t], dim=-1)
        
        # Step C: Predictions
        recon_img_t = decoder(latent_t)
        pred_rew_t = reward_predictor(latent_t)
        
        # Step D: Calculate losses at time t
        recon_loss += nn.functional.mse_loss(recon_img_t, img_t)
        reward_loss += nn.functional.mse_loss(pred_rew_t, rew_t)
        
        # KL Divergence: analytical distance between two Normal distributions
        # We use .clamp(min=1.0) or free-bits in Dreamer, but standard KL works perfectly for now
        kl_step = kl_divergence(post_dist, prior_dist).mean()
        kl_loss += kl_step

    # Divide by sequence length to get mean step loss
    recon_loss /= seq_len
    reward_loss /= seq_len
    kl_loss /= seq_len
    
    # Total loss function (Dreamer V1 weights KL down slightly to prioritize reconstruction)
    total_loss = recon_loss + reward_loss + 0.5 * kl_loss
    
    # 5. Backward pass and Optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("\n--- Backward Pass Check ---")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"Reward Loss:         {reward_loss.item():.4f}")
    print(f"KL Divergence Loss:  {kl_loss.item():.4f}")
    print(f"Total Combined Loss: {total_loss.item():.4f}")
    print("Optimization step completed successfully!")

if __name__ == "__main__":
    train_world_model_step()