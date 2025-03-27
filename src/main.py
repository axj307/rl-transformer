# src/main.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.environments.double_integrator_env import DoubleIntegratorEnv
from src.models.transformer import TransformerControlNetwork
from src.agents.transformer_agent import TransformerControlAgent
from src.training.trainer import Trainer
from src.utils.visualization import visualize_multiple_trajectories

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Environment parameters - increased max_steps for better convergence
    env = DoubleIntegratorEnv(
        dt=0.1, 
        max_steps=50,  
        control_limit=4.0  # Sufficient control authority
    )
    
    # Model architecture
    model = TransformerControlNetwork(
        state_dim=2,
        action_dim=1,
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.1,  # Add dropout for better generalization
        max_seq_length=50  # Match with max_steps
    )
    
    # Agent parameters - More stable learning
    agent = TransformerControlAgent(
        model=model,
        lr=0.0001,  # Reduce from 0.0003 to 0.0001
        gamma=0.99,  # Higher discount factor for long-term planning
        memory_length=50  # Match with max_steps
    )
    
    # Training parameters
    trainer = Trainer(
        env=env,
        agent=agent,
        num_episodes=10000,  # DOUBLE training duration
        logging_interval=100
    )
    
    # Train the agent
    metrics = trainer.train()
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['state_errors'])
    plt.title('Final State Error')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show(block=False)
    
    # Visualize a trajectory with the trained policy
    visualize_trajectory(env, agent)
    
    # Test with multiple initial states
    visualize_multiple_trajectories(env, agent)

def visualize_trajectory(env, agent, initial_state=None):
    """Visualize a control trajectory"""
    state = env.reset(initial_state)
    agent.reset()
    
    states = [state]
    actions = []
    
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, _, done, _ = env.step(action)
        
        states.append(next_state)
        actions.append(action)
        state = next_state
    
    states = np.array(states)
    actions = np.array(actions)
    
    # Plot trajectory
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(states[:, 0], states[:, 1])
    plt.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], c='r', s=100, label='End')
    plt.scatter(0, 0, c='b', s=100, label='Target')
    plt.title('State Trajectory')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(len(states)), states[:, 0])
    plt.title('Position vs. Time')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(len(actions)), actions)
    plt.title('Control Input vs. Time')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory.png')
    plt.show()

if __name__ == "__main__":
    main()