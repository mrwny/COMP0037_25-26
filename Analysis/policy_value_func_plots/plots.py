import csv
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

theta_vals = []
sweep_vals = []
runtime_vals = []
error_vals = []

# Prep data
with open("Coursework_01/Analysis/policy_value_func_plots/q3e_results.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        theta_vals.append(float(row["theta"]))
        sweep_vals.append(int(row["max_eval_steps"]))
        runtime_vals.append(float(row["runtime_s"]))
        error_vals.append(float(row["max_abs_diff_v"]))

theta_vals = np.array(theta_vals)
sweep_vals = np.array(sweep_vals)
runtime_vals = np.array(runtime_vals)
error_vals = np.array(error_vals)

unique_theta = sorted(set(theta_vals))
unique_sweeps = sorted(set(sweep_vals))

# Plot 1: Runtime vs Maximum Policy Evaluation steps
plt.figure()
for theta in unique_theta:
    mask = theta_vals == theta
    sweeps = sweep_vals[mask]
    runtimes = runtime_vals[mask]
    order = np.argsort(sweeps)
    plt.plot(sweeps[order], runtimes[order], marker='o', label=f"θ={theta}")

plt.xlabel("Max number of Policy Evaluation steps per iteration")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Maximum Policy Evaluation steps")
plt.legend()
plt.show()


# Plot 2: 3D Scatter Plot
sns.set(style="whitegrid")
df = pd.read_csv("Coursework_01/Analysis/policy_value_func_plots/q3e_results.csv")
df["log_theta"] = np.log10(df["theta"])
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
s=100
sc = ax.scatter(
    df["log_theta"],
    df["max_eval_steps"],
    df["runtime_s"],
    c=df["max_abs_diff_v"],
    cmap="viridis",
    s=80
)

ax.set_xlabel("log10(Theta)")
ax.set_ylabel("Max number of Policy Evaluation steps per iteration")
ax.set_zlabel("Runtime (s)")
cbar = plt.colorbar(sc)
cbar.set_label("Maximum Absolute Difference from V_baseline")
plt.show()


# Plot 3: Runtime vs Value Function Error Trade-off
plt.figure(figsize=(7,5))
eps = 1e-12
error_plot = np.maximum(error_vals, eps)
plt.scatter(runtime_vals, error_plot, s=100)
plt.yscale("log")
plt.xlabel("Runtime (s)")
plt.ylabel("Max |V - V_base| (log scale)")
plt.title("Runtime vs Value Function Error Trade-off")

plt.tight_layout()
plt.show()