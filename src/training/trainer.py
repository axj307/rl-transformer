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
        
        # FIX: Use 'total_error' instead of 'state_error' which doesn't exist
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'total_reward': total_reward,
            'episode_length': len(states),
            'final_state_error': info['total_error']  # Changed from 'state_error' to 'total_error'
        }
    
    def train(self):
        """Main training loop with batch collection"""
        print("Starting training...")
        
        # Parameters for batch collection - INCREASE THESE
        batch_size = 128  # Larger batch size
        max_buffer_size = 4096  # More experience data
        
        # Storage for batch data
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        # Initialize losses dict to prevent reference before assignment
        losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'total_loss': 0.0}
        
        # Better curriculum learning
        for episode in range(1, self.num_episodes + 1):
            # More gradual difficulty scaling
            if episode < self.num_episodes * 0.1:
                difficulty = 0.3  # Start moderate 
            elif episode < self.num_episodes * 0.3:
                difficulty = 0.6  # Increase over time
            elif episode < self.num_episodes * 0.6:
                difficulty = 0.8
            else:
                difficulty = 1.0
            
            # Reset environment with current difficulty
            state = self.env.reset(difficulty=difficulty)
            
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
            
            # Store metrics
            self.rewards_history.append(trajectory['total_reward'])
            self.state_errors.append(trajectory['final_state_error'])
            self.episode_lengths.append(trajectory['episode_length'])
            
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
        return {
            'rewards': self.rewards_history,
            'state_errors': self.state_errors,
            'episode_lengths': self.episode_lengths
        }