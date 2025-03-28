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
    def __init__(
        self, 
        state_dim=2,           # Position and velocity
        action_dim=1,          # Acceleration
        hidden_dim=64,         # Size of embeddings/hidden layers
        num_heads=4,           # Number of attention heads
        num_layers=2,          # Number of transformer layers
        max_seq_length=20,     # Max length of state history
        control_limit=1.0,     # Add this parameter
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.control_limit = control_limit  # Save the control limit
        
        # Main components of the transformer model:

        # 1. State embedding layer: converts state vectors to higher dimensional representations
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # 2. Positional encoding: adds temporal information to the embedded states
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length)

        # 3. Transformer encoder: processes the sequence of embedded states
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 4. Policy head (actor): outputs mean of action distribution
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Outputs [-1, 1]
            # Match scaling to environment's control limit
            Lambda(lambda x: x * self.control_limit)  # Scale to match env control limit
        )

        # 5. Value head (critic): estimates state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.0)  # Start with higher log_std for more exploration
    
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
        embeddings = self.positional_encoding(embeddings)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(embeddings)
        
        # Use the last state representation for action and value
        final_representation = transformer_output[:, -1, :]
        
        # Get action mean and value estimate
        action_mean = self.policy_head(final_representation)
        value = self.value_head(final_representation)
        
        # Fixed action standard deviation
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
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