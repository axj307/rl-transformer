import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(metrics):
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
    plt.show()

def visualize_multiple_trajectories(env, agent, initial_states=None):
    """Visualize multiple trajectories from different initial states"""
    if initial_states is None:
        # Define a grid of initial states to test
        initial_states = [
            [1.0, 0.0],    # Position offset only
            [0.0, 0.5],    # Velocity offset only
            [0.5, 0.5],    # Both positive
            [-0.5, 0.5],   # Mixed signs
            [-0.5, -0.5],  # Both negative
            [0.0, 0.0],    # Already at target (should stay there)
            [1.0, -0.5],   # Position + negative velocity
            [-1.0, 0.5],   # Negative position + velocity
        ]
    
    plt.figure(figsize=(12, 10))
    
    # Phase portrait (position vs velocity)
    plt.subplot(2, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_states)))
    
    for i, initial_state in enumerate(initial_states):
        trajectory = simulate_trajectory(env, agent, initial_state)
        positions = [state[0] for state in trajectory['states']]
        velocities = [state[1] for state in trajectory['states']]
        
        # Plot trajectory with color
        plt.plot(positions, velocities, '-', color=colors[i], 
                 label=f"Initial: {initial_state}")
        
        # Mark start point with a green circle
        plt.scatter(positions[0], velocities[0], color='green', s=100, marker='o')
        
        # Mark end point with a blue diamond
        plt.scatter(positions[-1], velocities[-1], color='blue', s=100, marker='D')
    
    # Target point
    plt.plot(0, 0, 'r*', markersize=15, label="Target")
    
    # Add a legend for start/end markers
    plt.scatter([], [], color='green', s=100, marker='o', label="Start")
    plt.scatter([], [], color='blue', s=100, marker='D', label="End")
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Position vs time
    plt.subplot(2, 2, 2)
    for i, initial_state in enumerate(initial_states):
        trajectory = simulate_trajectory(env, agent, initial_state)
        plt.plot(
            range(len(trajectory['states'])),
            [state[0] for state in trajectory['states']], 
            '-', color=colors[i]
        )
    plt.axhline(y=0, color='r', linestyle='--')  # Target
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.grid(True)
    
    # Velocity vs time
    plt.subplot(2, 2, 3)
    for i, initial_state in enumerate(initial_states):
        trajectory = simulate_trajectory(env, agent, initial_state)
        plt.plot(
            range(len(trajectory['states'])),
            [state[1] for state in trajectory['states']], 
            '-', color=colors[i]
        )
    plt.axhline(y=0, color='r', linestyle='--')  # Target
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')
    plt.grid(True)
    
    # Control action vs time
    plt.subplot(2, 2, 4)
    for i, initial_state in enumerate(initial_states):
        trajectory = simulate_trajectory(env, agent, initial_state)
        plt.plot(
            range(len(trajectory['actions'])),
            trajectory['actions'], 
            '-', color=colors[i]
        )
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.title('Control Actions vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('multiple_trajectories.png')
    plt.close()  # Close the figure to free memory

def simulate_trajectory(env, agent, initial_state):
    """Simulate a trajectory from given initial state"""
    state = env.reset(initial_state=initial_state)
    agent.reset()
    
    states = [state]
    actions = []
    rewards = []
    
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True)  # Use deterministic policy for testing
        next_state, reward, done, _ = env.step(action)
        
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards
    }