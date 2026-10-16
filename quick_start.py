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

# Reset the environment (normalized from environment to obserbation)
obs, state = env.reset()

print("Initial observation:", obs)

# List to preserve the observation
concentrations = [obs[0]]
temperatures = [obs[1]]

# Proportional control parameter
Kp = 10.0
Tc_base = 298.0

# Simulation
for t in range(nsteps):
    # Denormalization the current concentration
    Ca = (obs[0] + 1) / 2 * (1.0 - 0.7) + 0.7

    # Current target concentration
    Ca_SP = SP['Ca'][t]

    # Control error
    error = Ca_SP - Ca

    # Proportional control
    Tc = Tc_base - Kp * error
    # Tc = 298 - 0.5 = 297.5

    # Control Variable Limits
    Tc = np.clip(Tc, 295, 302)

    # Control input
    action = np.array([Tc])

    obs, rew, done, term, info = env.step(action)

    # preserve the observation
    concentrations.append(obs[0])
    temperatures.append(obs[1])

    

    print(
        f"step={t+1}, "
        f"Ca={Ca:.4f}, "
        f"SP={Ca_SP:.4f}, "
        f"Tc={Tc:.2f}, "
        f"concentration={obs[0]:.4f}, "
        f"temperatures={obs[1]:.4f}"
    )

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