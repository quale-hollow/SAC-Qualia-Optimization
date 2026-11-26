from utils.plotting_utils import plot_tradeoff
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import os

# Usage: python plot_tradeoffs.py
# Plots final return (last episode) vs RQO for VPER (exponential and linear priorities, actor update only)
# and VWAU in CartPole-v1, Pendulum-v1, and HalfCheetah-v4, one panel per environment,
# saved to sac_plots/all_envs_tradeoff.pdf.

names = ["VPER (exponential)", "VPER (linear)", "VWAU"]
methods = ["VPER_actor_exp", "VPER_actor_relu", "VWAU"]
omegas = [["0.2", "0.4", "0.6", "0.8", "1.0"]] * len(methods)
save_dir = "sac_plots/"

envs = [
    {"title": "CartPole-v1", "path": "results/cartpole_results"},
    {"title": "Pendulum-v1", "path": "results/pendulum_results"},
    {"title": "HalfCheetah-v4", "path": "results/halfcheetah_results"},
]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})
plt.style.use("seaborn-v0_8-paper")
os.makedirs(save_dir, exist_ok=True)

width_in = 397.48499 / 72.27

fig, axes = plt.subplots(1, len(envs), figsize=(width_in, width_in * 0.28), constrained_layout=True)
for ax, env in zip(axes.flat, envs):
    plot_tradeoff(ax, env["path"], methods, names, omegas, env["title"], NUM_TRIALS=100)
for ax in axes.flat[1:]:
    ax.set_ylabel("")

# One-line legend; "Standard SAC" shows both its marker and its dashed return line
handles, labels = axes.flat[0].get_legend_handles_labels()
i = labels.index("Standard SAC")
dashed = plt.Line2D([], [], color="C0", linestyle="--", linewidth=0.6, alpha=0.6)
handles[i] = (handles[i], dashed)
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=len(labels), frameon=False, handletextpad=0.4, columnspacing=1.0,
           handlelength=1.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})

fig.savefig(f"{save_dir}/all_envs_tradeoff.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
