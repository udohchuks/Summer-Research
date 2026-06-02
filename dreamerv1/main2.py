# main.py
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_config
from models import RSSM, ConvEncoder, ConvDecoder, RewardPredictor, ActionActor, ValueCritic
from buffer import SequenceReplayBuffer

def test_imagination_pipeline():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Imagination Pipeline on: {device}")
    
    # 1. Instantiate all 6 modules
    encoder = ConvEncoder(config).to(device)
    rssm = RSSM(config).to(device)
    reward_predictor = RewardPredictor(config).to(device)
    actor = ActionActor(config).to(device)
    critic = ValueCritic(config).to(device)
    
    # 2. Setup a mock batch of features from a posterior world model run
    # In reality, these come from unrolling the world model on real data.
    batch_size = config.training.batch_size
    
    # Imagine starting from a collection of posterior states collected from the buffer
    h_start = torch.randn(batch_size, config.world_model.deterministic_dim, device=device)
    z_start = torch.randn(batch_size, config.world_model.stochastic_dim, device=device)
    
    print(f"Starting Imagination Batch Size: {batch_size}")
    
    # 3. Latent Imagination Loop
    imagination_horizon = 15 # Dreamer defaults to 15 steps of mental rollout
    
    # We will accumulate trajectories of states, actions, and predicted rewards
    imagined_h = [h_start]
    imagined_z = [z_start]
    imagined_actions = []
    imagined_rewards = []
    
    current_h = h_start
    current_z = z_start
    
    print(f"Rolling out {imagination_horizon} steps into the mental future...")
    for t in range(imagination_horizon):
        # Combine state to feed into Actor
        current_latent = torch.cat([current_h, current_z], dim=-1)
        
        # Action Actor selects an action based on the imagined state
        action, action_dist = actor(current_latent)
        
        # RSSM Prior predicts what happens next purely using dynamics math
        current_h, current_z, prior_dist = rssm.prior_step(current_z, action)
        
        # Reward Predictor guesses how good this imagined step was
        next_latent = torch.cat([current_h, current_z], dim=-1)
        pred_reward = reward_predictor(next_latent)
        
        # Save values
        imagined_h.append(current_h)
        imagined_z.append(current_z)
        imagined_actions.append(action)
        imagined_rewards.append(pred_reward)
        
    # Stack lists into clean tensors for Actor-Critic loss calculation
    # Dim 0 will be the Time dimension of imagination, Dim 1 will be Batch
    img_actions_tensor = torch.stack(imagined_actions) # Shape: [15, 16, 3]
    img_rewards_tensor = torch.stack(imagined_rewards) # Shape: [15, 16, 1]
    
    print("\n--- Imagination Rollout Complete ---")
    print(f"Imagined Actions Trajectory Shape: {img_actions_tensor.shape}")
    print(f"Imagined Rewards Trajectory Shape: {img_rewards_tensor.shape}")
    print("--- IMAGINATION PIPELINE OK ---")

if __name__ == "__main__":
    test_imagination_pipeline()