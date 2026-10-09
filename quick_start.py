import pcgym
import numpy as np

# Simulation variables
nsteps = 100
T = 25

# Setpoint
SP = {'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))]} 

# Action and observation Space
action_space = {'low': np.array([295]), 'high': np.array([302])}
observation_space = {'low': np.array([0.7,300,0.8]),'high': np.array([1,350,0.9])}

# Construct the environment parameter dictionary
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP' :SP, 
    'o_space' : observation_space, 
    'a_space' : action_space, 
    'x0': np.array([0.8, 330, 0.8]), # Initial conditions [Ca, T, Ca_SP]
    'model': 'cstr', # Select the model
}

# Create environment
env = pcgym.make_env(env_params)

# Reset the environment
obs, state = env.reset()

print("Initial observation:", obs)

# Advance 10 steps with a random control input
for t in range(10):
    action = env.action_space.sample()

    obs, rew, done, term, info = env.step(action)

    print(f"step {t+1}: observation={obs}, reward={rew}")

    if done or term:
        break

env.close()