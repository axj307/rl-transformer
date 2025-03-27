# src/agents/transformer_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

class TransformerControlAgent:
    def __init__(
        self,
        model,
        lr=0.0003,
        gamma=0.99,
        memory_length=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory_length = memory_length
        
        # State history buffer
        self.state_history = deque(maxlen=memory_length)
        
        # Empty state (for padding)
        self.empty_state = np.zeros(model.state_dim)
        
        # State normalizer
        self.state_normalizer = StateNormalizer(model.state_dim)
    
    def reset(self):
        """Reset agent's state history"""
        self.state_history.clear()
    
    def get_state_history_tensor(self, state):
        """Convert state history to tensor for model input"""
        # Add current state to history
        self.state_history.append(state)
        
        # Pad with empty states if needed
        if len(self.state_history) < self.memory_length:
            padding = [self.empty_state] * (self.memory_length - len(self.state_history))
            padded_history = padding + list(self.state_history)
        else:
            padded_history = list(self.state_history)
        
        # Convert to tensor - fix slow operation by using numpy first
        padded_history_array = np.array(padded_history)
        state_tensor = torch.FloatTensor(padded_history_array).to(self.device)
        
        return state_tensor.unsqueeze(0)  # Add batch dimension
    
    def select_action(self, state, deterministic=False):
        """Select action based on current state and history"""
        normalized_state = self.state_normalizer.normalize(state)
        with torch.no_grad():
            state_tensor = self.get_state_history_tensor(normalized_state)
            action = self.model.get_action(state_tensor, deterministic)
            
        return action.cpu().numpy().flatten()
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update the model using PPO algorithm"""
        # First update the normalizer with flat states
        self.state_normalizer.update(states)
        
        # Normalize all states
        normalized_states = np.array([self.state_normalizer.normalize(s) for s in states])
        
        # Process data to create proper sequence inputs for transformer
        sequence_length = min(self.memory_length, len(states))
        batch_size = max(1, len(states) // sequence_length)
        
        # Create batches of sequences
        batch_states = []
        batch_actions = []
        batch_returns = []
        
        # Compute returns (discounted rewards)
        returns = []
        running_return = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            running_return = reward + self.gamma * running_return * (1 - done)
            returns.insert(0, running_return)
        
        # Create sequences for the transformer
        for i in range(0, len(normalized_states) - sequence_length + 1, sequence_length):
            if i + sequence_length <= len(normalized_states):
                seq_states = normalized_states[i:i+sequence_length]
                seq_actions = actions[i:i+sequence_length]
                seq_returns = returns[i:i+sequence_length]
                
                batch_states.append(seq_states)
                batch_actions.append(seq_actions)
                batch_returns.append(seq_returns)
        
        if not batch_states:  # If no complete sequences, use partial
            batch_states = [normalized_states[-sequence_length:]]
            batch_actions = [actions[-sequence_length:]]
            batch_returns = [returns[-sequence_length:]]
        
        # Convert to tensors - Fix slow tensor creation warning
        batch_states_np = np.array(batch_states)
        batch_actions_np = np.array(batch_actions)
        batch_returns_np = np.array(batch_returns)
        
        states_tensor = torch.FloatTensor(batch_states_np).to(self.device)
        actions_tensor = torch.FloatTensor(batch_actions_np).to(self.device)
        returns_tensor = torch.FloatTensor(batch_returns_np).to(self.device)
        
        # Make sure actions tensor has the right shape [batch, seq_len, action_dim]
        if actions_tensor.dim() == 2:
            actions_tensor = actions_tensor.unsqueeze(-1)
            
        # Get old action distributions and values BEFORE update
        with torch.no_grad():
            old_action_means, old_action_stds, old_values = self.model(states_tensor)
            
            # Ensure action means and stds have correct dimensions
            if old_action_means.dim() == 2:
                old_action_means = old_action_means.unsqueeze(-1)
            if old_action_stds.dim() == 2:
                old_action_stds = old_action_stds.unsqueeze(-1)
                
            old_dist = torch.distributions.Normal(old_action_means, old_action_stds)
            old_log_probs = old_dist.log_prob(actions_tensor)
            
            # Sum over action dimensions, not sequence dimension
            old_log_probs = old_log_probs.sum(dim=-1)
        
        # PPO typically uses multiple optimization epochs
        for _ in range(4):  # 4 update epochs per batch
            # Get current action distributions and values
            action_means, action_stds, values = self.model(states_tensor)
            
            # Ensure action means and stds have correct dimensions
            if action_means.dim() == 2:
                action_means = action_means.unsqueeze(-1)
            if action_stds.dim() == 2:
                action_stds = action_stds.unsqueeze(-1)
            
            # FIX: Properly handle the values tensor shape for sequence data
            if values.dim() == 2:  # [batch_size, 1]
                # Expand values to match returns_tensor shape [batch_size, seq_length]
                seq_length = returns_tensor.size(1)
                values = values.expand(-1, seq_length)
            
            # Calculate advantages using properly shaped values
            advantages = returns_tensor - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute action log probabilities
            dist = torch.distributions.Normal(action_means, action_stds)
            log_probs = dist.log_prob(actions_tensor)
            
            # Sum over action dimensions, not sequence dimension
            log_probs = log_probs.sum(dim=-1)
            
            # *** PPO CLIPPING - CRITICAL ADDITION ***
            ratio = torch.exp(log_probs - old_log_probs)
            clip_param = 0.2
            clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # FIX: Reshape returns_tensor for MSE loss calculation
            # Flatten both tensors to ensure they have the same shape
            value_loss = F.mse_loss(values.reshape(-1), returns_tensor.reshape(-1))
            
            # Entropy loss (for exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

class StateNormalizer:
    def __init__(self, state_dim):
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.n = 0
        
    def update(self, states):
        batch_size = states.shape[0]
        self.n += batch_size
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        
        # Update running statistics using Welford's algorithm
        delta = batch_mean - self.state_mean
        self.state_mean += delta * batch_size / self.n
        self.state_std = np.sqrt(
            (self.state_std**2 * (self.n - batch_size) + 
             batch_var * batch_size + 
             delta**2 * batch_size * (self.n - batch_size) / self.n) / self.n
        )
        
    def normalize(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

# The following commented code describes key agent functionalities but should not be executed
"""
Key agent functionalities:

1. State history management: maintains a buffer of past states
   self.state_history = deque(maxlen=memory_length)

2. Action selection: uses the transformer to select actions
   action = self.model.get_action(state_tensor, deterministic)

3. Learning algorithm: simplified PPO update
   - Computes returns from rewards
   - Computes advantages (returns - values)
   - Policy loss (maximize advantage * log_prob)
"""