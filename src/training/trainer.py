# src/training/trainer.py
import numpy as np
import torch
import time
from collections import deque

class Trainer:
    def __init__(self, env, agent, num_episodes=1000, logging_interval=10):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.logging_interval = logging_interval
        
        # Metrics
        self.rewards_history = []
        self.state_errors = []
        self.episode_lengths = []
    
    def collect_trajectory(self, verbose=False):
        """Collect a single episode trajectory"""
        state = self.env.reset()
        self.agent.reset()
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            action = self.agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Print detailed step info if verbose
            if verbose and (step % 10 == 0):  # Print every 10 steps to avoid too much output
                print(f"    Step {step}: State: {state}, Action: {action}, Reward: {reward:.2f}")
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            state = next_state
            total_reward += reward
            step += 1
        
        if verbose:
            print(f"    Episode complete - Final state: {state}, Total reward: {total_reward:.2f}")
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'total_reward': total_reward,
            'episode_length': len(states),
            'final_state_error': info['state_error']
        }
    
    def train(self):
        """Main training loop with batch collection"""
        print("Starting training...")
        
        # Parameters for batch collection - INCREASE THESE
        batch_size = 1024  # Increase from 64 to 128
        max_buffer_size = 4096  # Increase from 1024 to 4096
        
        # Storage for batch data
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Initialize losses dict to store latest losses
        losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Initialize metrics dictionaries
        metrics = {
            'rewards': [],
            'state_errors': [],
            'episode_lengths': []
        }
        
        # Add a metrics recording interval
        metrics_interval = 10  # Only record every 10th episode
        
        # Track recent average performance for curriculum advancement
        success_threshold = 0.03  # More lenient success threshold
        window_size = 10
        recent_errors = deque(maxlen=window_size)
        current_difficulty = 0.005  # Start easier
        
        for episode in range(1, self.num_episodes + 1):
            # Smooth curriculum based on recent performance
            if len(recent_errors) == window_size:
                success_rate = sum(1 for err in recent_errors if err < success_threshold) / window_size
                
                # Gentler difficulty increases
                if success_rate > 0.7 and current_difficulty < 1.0:  # Lower success threshold
                    current_difficulty = min(current_difficulty * 1.1, 1.0)  # Only 10% increase
                
                # Add difficulty decrease mechanism
                elif success_rate < 0.3 and current_difficulty > 0.005:
                    current_difficulty = max(current_difficulty * 0.9, 0.005)  # Decrease by 10%
            
            # Reset environment with current difficulty
            state = self.env.reset(difficulty=current_difficulty)
            
            # Print initial state for debug
            if episode % self.logging_interval == 0:
                print(f"Episode {episode} initial state: {state}")
            
            # Collect trajectory
            trajectory = self.collect_trajectory(verbose=(episode % self.logging_interval == 0))
            
            # Add to batch
            batch_states.extend(trajectory['states'])
            batch_actions.extend(trajectory['actions'])
            batch_rewards.extend(trajectory['rewards'])
            batch_next_states.extend(trajectory['next_states'])
            batch_dones.extend(trajectory['dones'])
            
            # Ensure buffer doesn't exceed max size
            if len(batch_states) > max_buffer_size:
                batch_states = batch_states[-max_buffer_size:]
                batch_actions = batch_actions[-max_buffer_size:]
                batch_rewards = batch_rewards[-max_buffer_size:]
                batch_next_states = batch_next_states[-max_buffer_size:]
                batch_dones = batch_dones[-max_buffer_size:]
            
            # Update only after collecting enough episodes
            if episode % batch_size == 0:
                losses = self.agent.update(
                    np.array(batch_states),
                    np.array(batch_actions),
                    np.array(batch_rewards),
                    np.array(batch_next_states),
                    np.array(batch_dones)
                )
                
                # Clear buffers after update (optional)
                # batch_states, batch_actions, batch_rewards = [], [], []
                # batch_next_states, batch_dones = [], []
            
            # Store metrics
            self.rewards_history.append(trajectory['total_reward'])
            self.state_errors.append(trajectory['final_state_error'])
            self.episode_lengths.append(trajectory['episode_length'])
            
            # Store recent errors for curriculum adjustment
            recent_errors.append(trajectory['final_state_error'])
            
            # Only store metrics every metrics_interval episodes
            if episode % metrics_interval == 0:
                metrics['rewards'].append(trajectory['total_reward'])
                metrics['state_errors'].append(trajectory['final_state_error'])
                metrics['episode_lengths'].append(trajectory['episode_length'])
            
            # Logging
            if episode % self.logging_interval == 0:
                avg_reward = np.mean(self.rewards_history[-self.logging_interval:])
                avg_error = np.mean(self.state_errors[-self.logging_interval:])
                
                print(f"Episode {episode}/{self.num_episodes}")
                print(f"  Initial state: {trajectory['states'][0]}")
                print(f"  Final state: {trajectory['states'][-1]}")
                print(f"  Actions: {trajectory['actions'].flatten()}")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Average state error: {avg_error:.4f}")
                print(f"  Average episode length: {np.mean(self.episode_lengths[-self.logging_interval:]):.1f}")
                print(f"  Policy loss: {losses['policy_loss']:.4f}")
                print(f"  Value loss: {losses['value_loss']:.4f}")
                print(f"  Entropy: {losses['entropy']:.4f}")
                print(f"  State normalizer mean: {self.agent.state_normalizer.state_mean}")
                print(f"  State normalizer std: {self.agent.state_normalizer.state_std}")
                print()
        
        print("Training complete!")
        return metrics