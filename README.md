# SAC-Qualia-Optimization

This repository contains Soft Actor-Critic (SAC) experiments for the paper **"Qualia Optimization in Reinforcement Learning: Balancing Agent Experience and Performance"**.

This repo includes:
- SAC experiments for `CartPole-v1`, `Pendulum-v1`, and `HalfCheetah-v4`
- implementations of rumination-correction methods: `VPER` and `VWAU`
- scripts to reproduce runs and generate paper plots/tables
- links to Zenodo-hosted SAC result data

## Repository Structure

```text
SAC-Qualia-Optimization/
├── results/                          # Extracted SAC result directories
├── (data downloaded from Zenodo)     # Place downloaded archives in repo root, then extract into results/
├── sac_implementations/
│   ├── SAC_Qualia_Continuous.py      # SAC for continuous actions (Pendulum, HalfCheetah)
│   ├── SAC_Qualia_Discrete.py        # SAC for discrete actions (CartPole, Atari support in code)
│   ├── buffers.py                    # Replay + prioritized replay buffers
│   └── atari_wrappers.py             # For Pong experiments, unused
├── utils/
│   └── plotting_utils.py             # Plotting + summary table utilities
├── sac_plots/                        # Generated PDFs
├── sac_experiment.py                 # Run a single SAC setting
├── plot.py                           # Generate paper plots + print LaTeX table rows
└── README.md
```

## Methods

The SAC interventions in this repo are:
- `control`: standard SAC baseline
- `VPER_actor_relu`: valence-prioritized experience replay (ReLU/linear valence prioritization), actor update prioritized
- `VPER_actor_exp`: valence-prioritized experience replay (exponential valence prioritization), actor update prioritized
- `VPER_both_relu`: valence-prioritized experience replay (ReLU/linear valence prioritization), actor and critic prioritized
- `VPER_both_exp`: valence-prioritized experience replay (exponential valence prioritization), actor and critic prioritized
- `VWAU`: valence-weighted actor updates

`omega` controls the strength of the qualia optimization mechanism.

## Notation

This README and code follow the paper's problem-setting notation:
- `B_k`: batch used at policy update `k`
- `Theta_{k-1}, Theta_k`: policy parameters before and after update `k`
- `Q_k`: per-update qualia valence
- `RQO`: cumulative reinforcement-qualia objective over updates
- `omega`: qualia optimization strength parameter
- `delta_i`: TD error for transition `i`

Paper definition of per-update qualia valence:
- `Q_k = (1/|B_k|) * sum_{t in B_k} (pi(S_t, A_t, Theta_k) / pi(S_t, A_t, Theta_{k-1}) - 1)`

Paper definition of RQO:
- `RQO = E[sum_{k=1..K} Q_k]`

Empirical estimate used in this repo:
- Per trial: `RQO_trial = sum_k Q_k`
- Reported table value: mean of `RQO_trial` across trials (with 95% CI)

## Environment Setup

Use separate environments when needed (recommended).

- Core:
  - `python==3.12.8`
  - `torch==2.5.1` (CPU or CUDA build)
  - `numpy==2.2.2`
  - `gymnasium==0.29.1`
- HalfCheetah (`HalfCheetah-v4`) also needs MuJoCo dependencies:
  - `mujoco==3.2.7`
  - MuJoCo rendering backends as needed (`glfw`, `PyOpenGL`)
- Plotting:
  - `matplotlib==3.10.0`
  - `scipy==1.16.3`
  - LaTeX install for `text.usetex=True` (for example TeX Live)

Optional but useful:
- `psutil` (used by replay buffer utilities when available)

## Results Data

Large result archives are hosted on Zenodo.

- Zenodo record URL: `https://zenodo.org/records/18884071?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4YzZlYTY4LThhOGYtNDM4MC1hZTk5LTAyYjlhZWQzZjQxNiIsImRhdGEiOnt9LCJyYW5kb20iOiI1MWEwZTM4YjY1NTQzMmYwZmY3ODc0ZTYyM2VhMjhiMiJ9.QG6-BzDY3RizOZeEGJ7pbWrVFqItI8CGM_8yWAPFRv4eV4uYqGfTPGQREexHwLvqQEea8wEGN4FvdQ7Udq2aQA`

- Zenodo DOI: `10.5281/zenodo.18884071`

Download and extract:

```bash
# Example:
# wget -O results.zip "<Zenodo record URL>"
# unzip results.zip -d .
```

After extraction, `results/` should include environment-specific folders such as:
- `results/cartpole_results`
- `results/pendulum_results`
- `results/halfcheetah_results`

Each environment folder uses this structure:

```text
results/<env>_results/
  <method>/                           # control, VPER_*, VWAU
    omega_<omega>/
      <trial_id>.npz
```

Each `.npz` file contains:
- `returns`: shape `[T, 2]`, columns `[global_step, episodic_return]`
- `ratios`: shape `[K, 2]`, columns `[global_step, mean_policy_ratio]`

`ratios` are logged from replay-buffer actions using:
- `mean_policy_ratio_k = mean_i exp(log pi_new(a_i|s_i) - log pi_old(a_i|s_i))`

The per-update qualia valence curve (`Q_k`) is plotted as:
- `Q_k = mean_policy_ratio_k - 1`

Final RQO per trial is computed as:
- `sum_k (mean_policy_ratio_k - 1)`

## Metric and Logging Details

### How `Q_k` is computed and logged

For each actor update:
1. Compute `pre_update_log_probs = actor.get_log_probs(obs_batch, action_batch)`.
2. Run actor update(s).
3. Compute `post_update_log_probs = actor.get_log_probs(obs_batch, action_batch)`.
4. Compute `log_ratio = post_update_log_probs - pre_update_log_probs`.
5. Log `mean(exp(log_ratio))` as one `ratios` point at the current `global_step`.

The `actor.get_log_probs` function was added specifically to evaluate log-probabilities of replay-buffer actions under the current policy for this metric.

Important cadence notes:
- Continuous SAC logs one ratio point only when actor updates run (`global_step % policy_frequency == 0`, after `learning_starts`).
- Discrete SAC logs one ratio point only on update steps (`global_step % update_frequency == 0`, after `learning_starts`).
- Therefore `K` (ratio rows) is generally different from `T` (episodic return rows).

Numerical detail:
- Continuous SAC clamps `log_ratio` to `[-10, 10]` before exponentiating.
- Discrete SAC does not currently clamp `log_ratio`.

### Method-specific meaning of `omega`

- `VPER_*`: `omega` is passed to prioritized replay as prioritization exponent (`alpha`), i.e., effective priority update is `p_i <- (valence_i)^alpha`.
- `VWAU`: `omega` scales TD error before softmax attention for actor loss weighting:
  - `w_i = softmax(omega * td_i)`.
- `control`: ignores `omega` and runs with `qualia_omega = 0.0`.

Valence transform used before VPER priority updates:
- `*_relu`: `valence = relu(td_error) + eps`
- `*_exp`: `valence = exp(td_error)` (with implementation-side cap for stability)

### Alignment with paper methodology

- Rumination correction in SAC is indirect: it changes replay sampling/weighting rather than directly modifying the SAC actor objective.
- Main paper SAC methodology corresponds to actor-only VPER:
  - actor batch is prioritized by valence;
  - critic batch remains uniform;
  - actor update intentionally omits importance-sampling correction.
- This repo also includes `VPER_both_*` variants (critic-prioritized ablations); when critic prioritization is enabled, critic loss uses IS weights.
- Episodic return logging is undiscounted episode return from the environment's episode statistics (`gamma = 1` style reporting in the paper tables/figures).

## Run Single Experiments

```bash
python sac_experiment.py [environment] [results_directory] [method] [omega] [num_trials]
```

Arguments:
- `environment`: `CartPole-v1`, `Pendulum-v1`, or `HalfCheetah-v4`
- `results_directory`: base output directory for this run
- `method`: one of `control`, `VPER_actor_relu`, `VPER_actor_exp`, `VPER_both_relu`, `VPER_both_exp`, `VWAU`
- `omega`: float (`0.0` for `control`)
- `num_trials`: integer

Examples:

```bash
python sac_experiment.py CartPole-v1 results/cartpole_results control 0.0 10
python sac_experiment.py CartPole-v1 results/cartpole_results VPER_actor_exp 0.4 10
python sac_experiment.py HalfCheetah-v4 results/halfcheetah_results VWAU 0.8 10
```

Reproducibility note:
- Trial seeds are randomly generated per invocation of `sac_experiment.py`.
- If exact reruns are required, store/report seed lists for each submission run.



## Plotting

Generate plots and print LaTeX table rows:

```bash
python plot.py [halfcheetah|cartpole|pendulum|all]
```

If no argument is provided, default is `all`.

Generated PDFs are written to:
- `sac_plots/cartpole/`
- `sac_plots/pendulum/`
- `sac_plots/halfcheetah/`

Plot/statistics details (`utils/plotting_utils.py`):
- 95% confidence intervals use Student's t distribution:
  - `CI_95 half-width = t.ppf(0.975, df=n-1) * (sample_std(ddof=1) / sqrt(n))`
- Time-step curves are built by:
  - concatenating all trial rows for a metric,
  - sorting by global step,
  - binning into `NUM_BINS` contiguous bins,
  - plotting bin mean with t-based 95% CI.
- For `ratios`, plotting converts to per-update qualia valence via:
  - `Q_k = ratio_k - 1`
- Final summary metrics used in tables:
  - Final return per trial: last logged episodic return (`ret[-1, 1]`)
  - Final RQO per trial: `sum_k (ratio_k - 1)`

