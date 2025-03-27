# src/environments/double_integrator_env.py
import numpy as np
from scipy.integrate import solve_ivp

class DoubleIntegratorEnv:
    def __init__(self, dt=0.1, max_steps=15, control_limit=1.0):
        self.dt = dt
        self.max_steps = max_steps
        self.control_limit = control_limit
        self.state = None
        self.step_count = 0
        self.reset()
        
    def reset(self, initial_state=None, difficulty=1.0):
        """Reset with better initial distribution"""
        if initial_state is None:
            # Use a grid of initial states rather than random
            # This helps ensure better coverage of the state space
            grid_points = np.linspace(-1.0, 1.0, 5) * difficulty
            pos_idx = np.random.randint(0, len(grid_points))
            vel_idx = np.random.randint(0, len(grid_points))
            
            self.state = np.array([
                grid_points[pos_idx],
                grid_points[vel_idx] * 0.5  # Velocity range is half of position
            ])
        else:
            self.state = np.array(initial_state)
        
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action):
        # Clip action to control limits
        action = np.clip(action, -self.control_limit, self.control_limit)
        
        # Store current state
        current_position, current_velocity = self.state
        current_error = np.sqrt(current_position**2 + current_velocity**2)
        
        # Dynamics integration
        def dynamics(t, x):
            position, velocity = x
            return [velocity, action[0]]
        
        # Integrate dynamics
        solution = solve_ivp(
            dynamics, 
            [0, self.dt], 
            self.state,
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        # Get new state
        new_position, new_velocity = solution.y[:, -1]
        self.state = np.array([new_position, new_velocity])
        
        # Calculate new error
        new_error = np.sqrt(new_position**2 + new_velocity**2)
        
        # IMPROVED REWARD FUNCTION
        # 1. Strong penalty for distance from target
        position_penalty = -1.0 * new_position**2  # Changed from -5.0
        velocity_penalty = -1.0 * new_velocity**2  # Changed from -5.0
        
        # 2. Reward for improvement
        improvement = max(0, current_error - new_error) * 5.0  # Changed from 20.0
        
        # 3. Large bonus for reaching target
        target_bonus = 0.0
        if new_error < 0.05:
            target_bonus = 20.0  # Changed from 100.0
        elif new_error < 0.1:
            target_bonus = 10.0  # Changed from 50.0
        elif new_error < 0.2:
            target_bonus = 5.0   # Changed from 20.0
        
        # 4. Control effort penalty
        control_penalty = -0.1 * action[0]**2
        
        # Combined reward
        reward = position_penalty + velocity_penalty + improvement + target_bonus + control_penalty
        
        # Update step count and check termination
        self.step_count += 1
        done = (self.step_count >= self.max_steps) or (new_error < 0.01)
        
        # More information in info dict
        info = {
            'position_error': abs(new_position),
            'velocity_error': abs(new_velocity),
            'total_error': new_error,
            'improvement': current_error - new_error
        }
        
        return self.state, reward, done, info