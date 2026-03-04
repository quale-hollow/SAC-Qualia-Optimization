import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# ----------------------------
# 95% CI (t-based) half-width
# ----------------------------
def ci95_halfwidth(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n <= 1:
        return 0.0
    sem = np.std(vals, ddof=1) / np.sqrt(n)
    return t.ppf(0.975, df=n - 1) * sem


def load_all_stats(filepath, num_bins=100, max_trials=None, final_return_window=5):
    """
    Load all experiment stats (learning curves + final values for return & RQO)
    in a single pass over the .npz files in `filepath`.

    Returns a dict with:
        - 'ret_steps', 'ret_mean', 'ret_ci95'
        - 'rq_steps',  'rq_mean',  'rq_ci95'
        - 'mean_final_return', 'ci95_final_return'
        - 'mean_final_rqo',    'ci95_final_rqo'
        - 'num_trials'
    """
    returns_combined = np.empty((0, 2))
    reinforcement_combined = np.empty((0, 2))

    final_returns = []
    qualia_returns = []

    files = sorted(os.listdir(filepath))
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        full_path = os.path.join(filepath, fname)
        data = np.load(full_path)

        ret = data["returns"]   # shape [T, 2]
        ratios = data["ratios"] # shape [T, 2]

        # accumulate for learning curves
        returns_combined = np.vstack((returns_combined, ret))
        reinforcement_combined = np.vstack((reinforcement_combined, ratios))

        # final return for this trial
        final_returns.append(ret[-1, 1])

        # final RQO / qualia return for this trial
        ratios_clean = ratios[np.isfinite(ratios).all(axis=1)]
        qualia_experiences = ratios_clean[:, 1] - 1.0
        qualia_returns.append(np.sum(qualia_experiences))

    num_trials = len(final_returns)

    # ---- learning curves: environment return ----
    returns_combined = returns_combined[returns_combined[:, 0].argsort()]
    bin_size_ret = max(1, len(returns_combined) // num_bins)

    ret_steps = np.zeros(num_bins)
    ret_mean = np.zeros(num_bins)
    ret_ci95 = np.zeros(num_bins)

    for i in range(num_bins):
        start = i * bin_size_ret
        if start >= len(returns_combined):
            # pad if we run out of data (e.g., too many bins)
            ret_steps[i] = ret_steps[i - 1] if i > 0 else 0.0
            ret_mean[i] = ret_mean[i - 1] if i > 0 else 0.0
            ret_ci95[i] = 0.0
            continue

        end = len(returns_combined) if i == num_bins - 1 else min(len(returns_combined), (i + 1) * bin_size_ret)
        chunk = returns_combined[start:end]

        ret_steps[i] = np.mean(chunk[:, 0])
        vals = chunk[:, 1]
        ret_mean[i] = np.mean(vals)
        ret_ci95[i] = ci95_halfwidth(vals)

    # ---- learning curves: reinforcement / RQO per update ----
    reinforcement_combined = reinforcement_combined[np.isfinite(reinforcement_combined).all(axis=1)]
    reinforcement_combined = reinforcement_combined[reinforcement_combined[:, 0].argsort()]
    bin_size_rq = max(1, len(reinforcement_combined) // num_bins)

    rq_steps = np.zeros(num_bins)
    rq_mean = np.zeros(num_bins)
    rq_ci95 = np.zeros(num_bins)

    for i in range(num_bins):
        start = i * bin_size_rq
        if start >= len(reinforcement_combined):
            rq_steps[i] = rq_steps[i - 1] if i > 0 else 0.0
            rq_mean[i] = rq_mean[i - 1] if i > 0 else 0.0
            rq_ci95[i] = 0.0
            continue

        end = len(reinforcement_combined) if i == num_bins - 1 else min(len(reinforcement_combined), (i + 1) * bin_size_rq)
        chunk = reinforcement_combined[start:end]

        rq_steps[i] = np.mean(chunk[:, 0])
        vals = chunk[:, 1] - 1.0
        rq_mean[i] = np.mean(vals)
        rq_ci95[i] = ci95_halfwidth(vals)

    # ---- final return stats ----
    final_returns = np.asarray(final_returns, dtype=np.float64)
    mean_final_return = np.mean(final_returns)
    ci95_final_return = ci95_halfwidth(final_returns)

    # ---- final qualia/RQO stats ----
    qualia_returns = np.asarray(qualia_returns, dtype=np.float64)
    mean_final_rqo = np.mean(qualia_returns)
    ci95_final_rqo = ci95_halfwidth(qualia_returns)

    return {
        "ret_steps": ret_steps,
        "ret_mean": ret_mean,
        "ret_ci95": ret_ci95,
        "rq_steps": rq_steps,
        "rq_mean": rq_mean,
        "rq_ci95": rq_ci95,
        "mean_final_return": mean_final_return,
        "ci95_final_return": ci95_final_return,
        "mean_final_rqo": mean_final_rqo,
        "ci95_final_rqo": ci95_final_rqo,
        "num_trials": num_trials,
    }


# graphing functions from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1, subplots=(1, 1)):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def axes_labels(x, pos):
    if x >= 1e9:
        return x
    if x >= 1e6:
        return f"{x*1e-6:.0f}M"
    elif x >= 1e3:
        if x % 1000 == 0:
            return f"{x*1e-3:.0f}K"
        else:
            return f"{x*1e-3:.1f}K"
    elif x < 10:
        return f"{x:.2f}"
    else:
        return f"{x:.0f}"


def plot_curves(methods, names, omegas, line_colors, path, env,
                save_dir="plots", qe_lims=None, NUM_TRIALS=None, NUM_BINS=100):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    width = 397.48499
    inches_per_pt = 1 / 72.27

    fig_width = width * inches_per_pt
    fig_height = fig_width * (1 / 4)
    layout = (1, 4)
    fig_dim = (fig_width, fig_height)

    print("Plotting results for", env)

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }

    plt.rcParams.update(tex_fonts)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["xtick.major.pad"] = 0.5
    plt.rcParams["ytick.major.pad"] = 0.5

    # ---------- CONTROL / BASELINE STATS ----------
    control_stats = load_all_stats(
        f"{path}/control/omega_0.0/", num_bins=NUM_BINS, max_trials=NUM_TRIALS
    )
    print("Count for baseline:", control_stats["num_trials"])

    bl_ret_x = control_stats["ret_steps"]
    bl_ret_y = control_stats["ret_mean"]
    bl_ret_ci = control_stats["ret_ci95"]

    bl_rq_x = control_stats["rq_steps"]
    bl_rq_y = control_stats["rq_mean"]
    bl_rq_ci = control_stats["rq_ci95"]

    bl_final_return = control_stats["mean_final_return"]
    bl_final_return_ci = control_stats["ci95_final_return"]
    bl_final_rqo = control_stats["mean_final_rqo"]
    bl_final_rqo_ci = control_stats["ci95_final_rqo"]

    table1 = (
        "Standard SAC & N/A & "
        f"{bl_final_return:.2f}$\\pm${bl_final_return_ci:.2f} & "
        f"{bl_final_rqo:.2f}$\\pm${bl_final_rqo_ci:.2f} \\\\ \\midrule\n"
    )

    # ---------- METHODS ----------
    for i, method in enumerate(methods):
        control_omega = 0.0

        fig, ax = plt.subplots(
            layout[0], layout[1], figsize=fig_dim, constrained_layout=True
        )
        fig.suptitle(names[i], fontsize=9)
        label_pad = 0

        ax[0].set_xlabel("Time Step", labelpad=label_pad)
        ax[0].set_ylabel(r"Return", labelpad=label_pad)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[1].set_xlabel("Time Step", labelpad=label_pad)
        ax[1].set_ylabel(r"Per-Update $Q_k$", labelpad=label_pad)
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        if qe_lims is not None:
            ax[1].set_ylim(*qe_lims)

        ax[2].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[2].set_ylabel("Final Return", labelpad=label_pad)
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[3].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[3].set_ylabel("Final RQO", labelpad=label_pad)
        ax[3].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        # ---------- plot control data ----------
        control_suffix = " (standard SAC)"
        control_label = fr"$\omega={control_omega}$" if method != "VPER" else ""
        control_label += control_suffix

        ax[0].plot(bl_ret_x, bl_ret_y, label=control_label, color="C0")
        ax[0].fill_between(
            bl_ret_x,
            bl_ret_y - bl_ret_ci,
            bl_ret_y + bl_ret_ci,
            alpha=0.3,
            color="C0",
        )

        ax[1].plot(bl_rq_x, bl_rq_y, color="C0")
        ax[1].fill_between(
            bl_rq_x,
            bl_rq_y - bl_rq_ci,
            bl_rq_y + bl_rq_ci,
            alpha=0.3,
            color="C0",
        )

        ax[2].errorbar(control_omega, bl_final_return, yerr=bl_final_return_ci, fmt="o", color="C0")
        ax[3].errorbar(control_omega, bl_final_rqo, yerr=bl_final_rqo_ci, fmt="o", color="C0")

        # ---------- PLOT LEARNING CURVES FOR EACH OMEGA ----------
        for j, omega in enumerate(omegas[i]):
            stats = load_all_stats(
                f"{path}/{method}/omega_{omega}/",
                num_bins=NUM_BINS,
                max_trials=NUM_TRIALS,
            )
            print("Count for", method, omega, ":", stats["num_trials"])

            # Returns vs timestep
            x_ret = stats["ret_steps"]
            y_ret = stats["ret_mean"]
            ci_ret = stats["ret_ci95"]

            suffix = " (SAC w/ PER)" if (method == "VPER" and float(omega) == 0.0) else ""
            label = fr"$\omega={omega}$" + suffix

            ax[0].plot(x_ret, y_ret, label=label, color=line_colors[j])
            ax[0].fill_between(
                x_ret,
                y_ret - ci_ret,
                y_ret + ci_ret,
                alpha=0.3,
                color=line_colors[j],
            )

            # RQO / reinforcement vs timestep
            x_rq = stats["rq_steps"]
            y_rq = stats["rq_mean"]
            ci_rq = stats["rq_ci95"]

            ax[1].plot(x_rq, y_rq, color=line_colors[j])
            ax[1].fill_between(
                x_rq,
                y_rq - ci_rq,
                y_rq + ci_rq,
                alpha=0.3,
                color=line_colors[j],
            )

            # Final stats
            ret = stats["mean_final_return"]
            ret_ci = stats["ci95_final_return"]
            rqo = stats["mean_final_rqo"]
            rqo_ci = stats["ci95_final_rqo"]

            # add to tex table
            table1 += (f"{names[i]} & " if j == 0 else "\t & ")
            table1 += (
                f"{omega} & {ret:.2f}$\\pm${ret_ci:.2f} & "
                f"{rqo:.2f}$\\pm${rqo_ci:.2f} \\\\"
            )
            if j == len(omegas[i]) - 1:
                table1 += "\\midrule"
            table1 += "\n"

            # scatter / errorbar vs omega
            ax[2].errorbar(float(omega), ret, yerr=ret_ci, fmt="o", color=line_colors[j])
            ax[3].errorbar(float(omega), rqo, yerr=rqo_ci, fmt="o", color=line_colors[j])

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            mode="expand",
            ncol=len(omegas[i]) + 1,
            columnspacing=1,
            handletextpad=0.2,
            borderpad=0,
            borderaxespad=0,
            frameon=False,
        )
        fig.get_layout_engine().set(h_pad=0.01)

        plt.savefig(f"{save_dir}/{env}_{method}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()

    print("\n\n")
    print(table1)