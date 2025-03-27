# src/models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs"""
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class TransformerControlNetwork(nn.Module):
    """Transformer-based actor-critic network for control problems"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_heads=8, 
                 num_layers=3, dropout=0.1, max_seq_length=100):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,  # Larger feedforward network
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Output heads with larger capacity
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.action_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),  # Use softplus instead of exp for better stability
            Lambda(lambda x: x + 1e-5)  # Add small constant to prevent zeros
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states):
        """
        Forward pass through the network
        
        Args:
            states: Tensor of shape [batch_size, seq_len, state_dim]
                   or [batch_size, state_dim] for single state
                   
        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            value: Estimated state value
        """
        # Handle single state input
        if len(states.shape) == 2:
            states = states.unsqueeze(1)  # Add sequence dimension
            
        batch_size, seq_len, _ = states.shape
        
        # Embed states
        embeddings = self.state_embedding(states)
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(embeddings)
        
        # Use the last state representation for action and value
        final_representation = transformer_output[:, -1, :]
        
        # Get action mean and value estimate
        action_mean = self.action_mean(final_representation)
        action_std = self.action_std(final_representation)
        value = self.value(final_representation)
        
        return action_mean, action_std, value
    
    def get_action(self, states, deterministic=False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(states)
        
        if deterministic:
            return action_mean
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(action_mean, action_std)
        action = normal.sample()
        
        return action