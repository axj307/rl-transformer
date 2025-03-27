class Config:
    """Default configuration for double integrator control"""
    
    # Environment parameters
    dt = 0.1
    max_steps = 100
    control_limit = 1.0
    
    # Model parameters
    state_dim = 2
    action_dim = 1
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    max_seq_length = 20
    
    # Agent parameters
    learning_rate = 0.0003
    gamma = 0.99
    memory_length = 20
    
    # Training parameters
    num_episodes = 500
    logging_interval = 10