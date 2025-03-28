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
    
    # Environment parameters
    control_limit = 2.0
    env = DoubleIntegratorEnv(dt=0.1, max_steps=15, control_limit=control_limit)   
    
    # Model architecture - pass control limit
    model = TransformerControlNetwork(
        state_dim=2,
        action_dim=1,
        hidden_dim=64,       # Increase from 32 back to 64
        num_heads=4,         # Increase from 2 back to 4
        num_layers=2,        # Increase from 1 back to 2
        max_seq_length=15,
        control_limit=control_limit  # Match the environment's control limit
    )
    
    # Learning parameters
    agent = TransformerControlAgent(
        model=model,
        lr=0.0003,            # Increase from 0.0001 back to 0.0003 for faster learning
        gamma=0.95,           # Slightly lower gamma to focus more on immediate rewards
        memory_length=15      # Match memory length to max_steps
    )
    
    # Training parameters
    trainer = Trainer(
        env=env,
        agent=agent,
        num_episodes=20000,    # Increase from 2000 to 5000
        logging_interval=100  # Change from 20 to 100 to reduce output volume
    )
    
    # Train the agent
    metrics = trainer.train()
    
    # Create x-axis values that match the recorded intervals
    metrics_interval = 10  # Same value as used in the Trainer class
    episodes = [i * metrics_interval for i in range(len(metrics['rewards']))]
    
    # Plot training progress with fewer points
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episodes, metrics['rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(episodes, metrics['state_errors'])
    plt.title('Final State Error')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    
    plt.subplot(1, 3, 3)
    plt.plot(episodes, metrics['episode_lengths'])
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()  # Close the figure to free memory
    
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
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    main()