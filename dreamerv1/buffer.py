import numpy as np
import torch

class SequenceReplayBuffer:
    def __init__(self, config):
        self.capacity = config.training.buffer_capacity
        self.seq_len = config.training.sequence_length
        self.batch_size = config.training.batch_size
        
        # Determine shapes from config
        self.img_shape = config.environment.image_shape
        self.action_dim = config.environment.action_dim
        
        # Pre-allocate numpy buffers for memory efficiency
        self.images = np.zeros((self.capacity, *self.img_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminals = np.zeros((self.capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.size = 0

    def add(self, image, action, reward, terminal):
        """Adds a single environment step transition."""
        self.images[self.idx] = image
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = terminal
        
        # Circular buffer pointer management
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, device):
        """
        Samples a batch of sequential segments.
        Returns tensors of shape: [Batch Size, Sequence Length, Feature Dimensions]
        """
        batch_images = []
        batch_actions = []
        batch_rewards = []
        batch_terminals = []
        
        # We need to make sure the sequence doesn't wrap around the end edge of the circular buffer
        # and doesn't overshoot the current total size of stored elements.
        valid_range = self.size - self.seq_len
        
        for _ in range(self.batch_size):
            start_idx = np.random.randint(0, valid_range)
            end_idx = start_idx + self.seq_len
            
            batch_images.append(self.images[start_idx:end_idx])
            batch_actions.append(self.actions[start_idx:end_idx])
            batch_rewards.append(self.rewards[start_idx:end_idx])
            batch_terminals.append(self.terminals[start_idx:end_idx])
            
        # Convert list of sequences into numpy stacks -> then move to torch tensors
        images_t = torch.tensor(np.stack(batch_images), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.stack(batch_actions), dtype=torch.float32, device=device)
        rewards_t = torch.tensor(np.stack(batch_rewards), dtype=torch.float32, device=device)
        terminals_t = torch.tensor(np.stack(batch_terminals), dtype=torch.float32, device=device)
        
        return images_t, actions_t, rewards_t, terminals_t