from pcgym import make_env
import pcgym
import numpy as np
from stable_baselines3 import PPO
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Simulation variables
nsteps = 10
T = 26

# Setpoint
SP = {'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))]} 

# Action and observation Space
action_space = {'low': np.array([295]), 'high': np.array([302])}
observation_space = {'low': np.array([0.7,300,0.8]),'high': np.array([1,350,0.9])}

r_scale ={
    'Ca': 1e3 #Reward scale for each state
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, # Setpoint
    'o_space' : observation_space, # Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([0.8,330,0.8]), # Initial conditions 
    'model': 'cstr', # Select the model
    'r_scale': r_scale, # Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': True, # Normalise the actions
    'normalise_o':True, # Normalise the states,
    'noise':True, # Add noise to the states
    'integration_method': 'casadi', # Select the integration method
    'noise_percentage':0.001 # Noise percentage
}
env = make_env(env_params)

# Policy Training
nsteps_learning = 3e4
PPO_policy = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001).learn(nsteps_learning)

# Policy Rollout and Plotting
evaluator, data = env.plot_rollout({'PPO': PPO_policy}, reps = 10)