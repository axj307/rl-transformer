# src/environments/double_integrator_env.py
import numpy as np
from scipy.integrate import solve_ivp

class DoubleIntegratorEnv:
    def __init__(self, dt=0.1, max_steps=15, control_limit=1.0):
        self.dt = dt
        self.max_steps = max_steps
        self.control_limit = control_limit
        # Add domain size parameters for easier modification
        self.position_range = 0.5  # Changed from 1.0 to 0.5
        self.velocity_range = 0.25  # Changed from 0.5 to 0.25
        self.state = None
        self.step_count = 0
        self.reset()
        
    def reset(self, initial_state=None, difficulty=1.0):
        """Reset with better initial distribution"""
        if initial_state is None:
            # Use a grid of initial states rather than random
            # This helps ensure better coverage of the state space
            # Changed from -1.0, 1.0 to -0.5, 0.5
            grid_points = np.linspace(-self.position_range, self.position_range, 5) * difficulty
            pos_idx = np.random.randint(0, len(grid_points))   
            vel_idx = np.random.randint(0, len(grid_points))
            
            self.state = np.array([
                grid_points[pos_idx],
                grid_points[vel_idx] * self.velocity_range  # Velocity range is now parameterized
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
        
        # Store previous error for progress reward
        old_error = np.sqrt(position**2 + velocity**2)
        
        # Calculate new error
        new_error = np.sqrt(new_position**2 + new_velocity**2)
        
        # IMPROVED REWARD FUNCTION
        # Stronger progress reward with quadratic scaling (rewards more for larger improvements)
        progress_reward = 10.0 * (old_error - new_error) * (1 + 5.0 * old_error)
        
        # Earlier proximity bonus that scales smoothly
        proximity_bonus = 10.0 * np.exp(-20.0 * new_error)  # Exponential scaling
        
        # Decrease distance penalty slightly to avoid overly conservative behavior
        distance_penalty = -1.0 * new_error
        
        # Control penalty remains the same
        control_penalty = -0.1 * (action**2)
        
        # Combined reward
        reward = progress_reward + proximity_bonus + distance_penalty + control_penalty
        
        # Early termination when goal reached
        goal_reached = new_error < 0.03
        done = (self.step_count >= self.max_steps) or goal_reached
        
        # Terminal state reward
        if goal_reached:
            reward += 50.0  # Bigger fixed bonus for reaching goal
        
        # State error (for evaluation)
        state_error = new_error
        
        return self.state.copy(), reward, done, {"state_error": state_error}        
    
    
    
            # # Calculate error
        # new_error = np.sqrt(new_position**2 + new_velocity**2)
        
        # # SIMPLIFIED REWARD FUNCTION
        # # Strong negative quadratic penalty based on distance from target
        # distance_penalty = -10.0 * (new_position**2 + new_velocity**2)
        
        # # Large bonus for being close to target
        # completion_bonus = 50.0 if new_error < 0.05 else 0.0
        
        # # Smaller control penalty to discourage excessive control
        # control_penalty = -0.1 * (action**2)
        
        # # Combined reward
        # reward = distance_penalty + completion_bonus + control_penalty
        
        # # Check if done
        # done = self.step_count >= self.max_steps
        
        # # State error (for evaluation)
        # state_error = np.sqrt(new_position**2 + new_velocity**2)
        
        # return self.state.copy(), reward, done, {"state_error": state_error}