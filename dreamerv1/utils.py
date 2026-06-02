# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import load_config, compute_lambda_returns
from models import RSSM, ConvEncoder, ConvDecoder, RewardPredictor, ActionActor, ValueCritic
from buffer import SequenceReplayBuffer

def complete_dreamer_step():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Complete Dreamer Cycle on {device} ===")
    
    # 1. Initialize Networks
    encoder = ConvEncoder(config).to(device)
    decoder = ConvDecoder(config).to(device)
    rssm = RSSM(config).to(device)
    reward_predictor = RewardPredictor(config).to(device)
    actor = ActionActor(config).to(device)
    critic = ValueCritic(config).to(device)
    
    # 2. Setup Optimizers
    wm_params = (list(encoder.parameters()) + list(decoder.parameters()) + 
                 list(rssm.parameters()) + list(reward_predictor.parameters()))
    wm_optimizer = optim.Adam(wm_params, lr=3e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.training.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.training.critic_lr)
    
    # 3. Fill buffer with fake data for structural execution
    buffer = SequenceReplayBuffer(config)
    for _ in range(100):
        buffer.add(
            torch.randn(*config.environment.image_shape).numpy(),
            torch.randn(config.environment.action_dim).numpy(),
            [1.0], [0.0]
        )
        
    # ==========================================
    # PHASE 1: Train World Model
    # ==========================================
    obs_seq, act_seq, rew_seq, _ = buffer.sample(device)
    batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
    
    h_t = torch.zeros(batch_size, config.world_model.deterministic_dim, device=device)
    z_t = torch.zeros(batch_size, config.world_model.stochastic_dim, device=device)
    
    # Keep track of collected posteriors to seed imagination later
    posterior_h_states = []
    posterior_z_states = []
    
    wm_loss = 0.0
    for t in range(seq_len):
        h_t, _, _ = rssm.prior_step(z_t, act_seq[:, t])
        embed_t = encoder(obs_seq[:, t])
        z_t, _ = rssm.posterior_step(h_t, embed_t)
        
        # Save belief states (detach them so gradients don't leak into Phase 1 during Phase 2)
        posterior_h_states.append(h_t.detach())
        posterior_z_states.append(z_t.detach())
        
        latent_t = torch.cat([h_t, z_t], dim=-1)
        recon = decoder(latent_t)
        pred_rew = reward_predictor(latent_t)
        
        wm_loss += nn.functional.mse_loss(recon, obs_seq[:, t]) + nn.functional.mse_loss(pred_rew, rew_seq[:, t])
        
    wm_optimizer.zero_grad()
    wm_loss.backward()
    wm_optimizer.step()
    print(f"-> World Model Loss updated: {wm_loss.item()/seq_len:.4f}")
    
    # ==========================================
    # PHASE 2: Latent Imagination (Actor-Critic)
    # ==========================================
    # Flatten collected states across batch and time dimensions to create a massive parallel start pool
    flat_h = torch.stack(posterior_h_states).view(-1, config.world_model.deterministic_dim)
    flat_z = torch.stack(posterior_z_states).view(-1, config.world_model.stochastic_dim)
    
    # Sub-sample to keep batch sizes reasonable for local PC
    sample_indices = torch.randperm(flat_h.shape[0])[:config.training.batch_size]
    curr_h = flat_h[sample_indices]
    curr_z = flat_z[sample_indices]
    
    imagined_latents = []
    imagined_rewards = []
    imagined_log_probs = []
    
    # Unroll into the future
    for t in range(config.training.imagination_horizon):
        curr_latent = torch.cat([curr_h, curr_z], dim=-1)
        imagined_latents.append(curr_latent)
        
        action, action_dist = actor(curr_latent)
        imagined_log_probs.append(action_dist.log_prob(action).sum(dim=-1, keepdim=True))
        
        curr_h, curr_z, _ = rssm.prior_step(curr_z, action)
        next_latent = torch.cat([curr_h, curr_z], dim=-1)
        imagined_rewards.append(reward_predictor(next_latent))
        
    # Append the last boundary state for the critic target
    imagined_latents.append(torch.cat([curr_h, curr_z], dim=-1))
    
    # Stack tensors
    img_latents = torch.stack(imagined_latents)      # [Horizon + 1, Batch, Latent_Dim]
    img_rewards = torch.stack(imagined_rewards)      # [Horizon, Batch, 1]
    img_log_probs = torch.stack(imagined_log_probs)  # [Horizon, Batch, 1]
    
    # Get Critic value predictions
    img_values = critic(img_latents)                 # [Horizon + 1, Batch, 1]
    
    # 4. Compute Targets
    lambda_targets = compute_lambda_returns(
        img_rewards, img_values, config.training.discount, config.training.lambda_
    ).detach() # Target values must be detached!
    
    # ==========================================
    # PHASE 3: Policy & Value Updates
    # ==========================================
    # Actor Loss: Maximize expected lambda returns (using REINFORCE or pathwise gradients)
    # Dreamer V1 primarily leverages analytical pathwise gradients since everything is differentiable!
    actor_loss = -lambda_targets.mean() 
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Critic Loss: Predict the lambda returns accurately
    critic_loss = nn.functional.mse_loss(img_values[:-1], lambda_targets)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    print(f"-> Actor Policy Loss:        {actor_loss.item():.4f}")
    print(f"-> Critic Value Loss:         {critic_loss.item():.4f}")
    print("=== FULL ITERATION CYCLE SUCCESSFUL ===")

if __name__ == "__main__":
    complete_dreamer_step()