import pcgym
import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

# List to preserve the observation
concentrations = [obs[0]]
temperatures = [obs[1]]

# Set a specific control variable
action = np.array([298])

# Simulation
for t in range(nsteps):
    obs, rew, done, term, info = env.step(action)

    # preserve the observation
    concentrations.append(obs[0])
    temperatures.append(obs[1])

    print(f"step {t+1}: observation={obs}, reward={rew}")

    if done or term:
        break

env.close()

# Timeline
time = np.arange(len(concentrations)) * (T / nsteps)

# Create a Graph
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Concentration Graph
axes[0].plot(time, concentrations, label="Concentrations Ca")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Concentration")
axes[0].legend()

# Humidity Graph
axes[1].plot(time, temperatures, label="Reactor Temperature")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Temperature [K]")
axes[1].set_title("Reactor Temperature")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("cstr_result.png", dpi=300, bbox_inches="tight")
plt.close()

# plt.show()