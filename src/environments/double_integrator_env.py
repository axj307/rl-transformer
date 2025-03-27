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
        """Take a step in the environment using control input"""
        # Extract scalar value if action is an array
        if hasattr(action, "__len__"):
            action = float(action[0])
        
        # Clip action to control limits
        action = np.clip(action, -self.control_limit, self.control_limit)
        
        # Unpack state
        position, velocity = self.state
        
        # Define the system dynamics
        def system_dynamics(t, state, acceleration):
            x, v = state
            return [v, acceleration]  # dx/dt = v, dv/dt = a
        
        # Solve using DOPRI method (RK45 in scipy)
        solution = solve_ivp(
            fun=lambda t, y: system_dynamics(t, y, action),
            t_span=[0, self.dt],
            y0=[position, velocity],
            method='RK45',  # This is the DOPRI45 method
            rtol=1e-6,
            atol=1e-6
        )
        
        # Extract the solution at the end time
        new_position = solution.y[0, -1]
        new_velocity = solution.y[1, -1]
        
        # Update state
        self.state = np.array([new_position, new_velocity])
        self.step_count += 1
        
        # Store current error before update
        current_error = np.sqrt(position**2 + velocity**2)
        
        # Calculate new error
        new_error = np.sqrt(new_position**2 + new_velocity**2)
        
        # Modified reward components
        base_reward = -(new_position**2 + new_velocity**2) * 0.5  # Increase base penalty
        improvement_reward = max(0, current_error - new_error) * 15.0  # Stronger improvement incentive
        completion_bonus = 50.0 if new_error < 0.05 else 0.0  # Much larger bonus for accuracy
        
        # Additional penalty for being far from target
        distance_penalty = -new_error * 0.3
        
        # Combined reward with distance penalty
        reward = base_reward + improvement_reward + completion_bonus + distance_penalty
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # State error (for evaluation)
        state_error = np.sqrt(new_position**2 + new_velocity**2)
        
        return self.state.copy(), reward, done, {"state_error": state_error}