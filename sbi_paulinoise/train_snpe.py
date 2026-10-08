"""Sequential NPE training (plain NLL)."""

import os
import torch
import numpy as np

import torch.nn as nn
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior

from .simulate import run_parallel
from .visualize import plot_training_loss


# ── density estimator factory ─────────────────────────────────────────────────

def _build_density_estimator(config: dict, n_obs: int | None = None):
    extra = {}
    if config.get("FLOW_MODEL") == "zuko_nsf":
        extra["num_bins"] = config.get("NSF_NUM_BINS", 8)
    if n_obs is not None:
        extra["embedding_net"] = nn.Identity()
    return posterior_nn(
        model=config.get("FLOW_MODEL", "zuko_nsf"),
        hidden_features=config["FLOW_HIDDEN_FEATURES"],
        num_transforms=config["NUM_TRANSFORMS"],
        **extra,
    )


def _build_posterior_for_estimator(inference, estimator, prior, device, x_obs_squeezed, config):
    """Build posterior from trained estimator using DirectPosterior.

    DirectPosterior samples directly from the normalising flow without any
    prior-based rejection loop.  This avoids the "0% acceptance rate" hang
    that occurs with RejectionPosterior when the posterior is concentrated
    near the prior boundary.
    """
    posterior = DirectPosterior(posterior_estimator=estimator, prior=prior, device=device)
    return posterior.set_default_x(x_obs_squeezed)


class PermissivePrior:
    """Dummy prior that returns log_prob=0; disables rejection sampling."""
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta.shape[0], device=theta.device)


def _ensure_samples_2d(samples: np.ndarray, num_params: int) -> np.ndarray:
    """Return posterior sample array as (n_samples, num_params)."""
    b = np.asarray(samples)
    if b.ndim == 2 and b.shape[1] != num_params and b.shape[0] == num_params:
        b = b.T
    elif b.ndim == 1 and b.size % num_params == 0:
        b = b.reshape(-1, num_params)
    if b.ndim != 2 or b.shape[1] != num_params:
        raise ValueError(f"posterior.sample() returned shape {samples.shape}; cannot get (n, {num_params})")
    return b


def _infer_n_qubits_from_num_params(num_params: int) -> int | None:
    """Mirror plot_scatter_inference() layout detection."""
    if num_params % 15 == 0:
        return int(num_params // 15 + 1)
    if (num_params + 15) % 21 == 0:
        return int((num_params + 15) // 21)
    return None


def _nogauge_keep_mask(samples: np.ndarray) -> np.ndarray:
    """Keep-mask matching the scatter's 'No gauge' panel (std_topk_from_nqubits).

    Drops the top-k highest-posterior-std parameters, k = 6*(n_qubits-2).
    """
    num_params = int(samples.shape[1])
    std = samples.std(axis=0)
    n_qubits = _infer_n_qubits_from_num_params(num_params)
    if n_qubits is None:
        return np.ones((num_params,), dtype=bool)
    k = max(0, min(int(6 * (n_qubits - 2)), num_params))
    keep = np.ones((num_params,), dtype=bool)
    if k == 0:
        return keep
    keep[np.argsort(-std)[:k]] = False
    return keep


def _r2_mean_vs_truth(samples: np.ndarray, truth: np.ndarray) -> float:
    """R^2 using posterior mean as predictor of the ground-truth parameters."""
    post_mean = samples.mean(axis=0)
    ss_tot = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((truth - post_mean) ** 2))
    return 1.0 - ss_res / ss_tot


# ── main train_snpe entry point ───────────────────────────────────────────────

def train_snpe(
    config: dict,
    x_obs: torch.Tensor,
    theta_truth: torch.Tensor | None,
    num_params: int,
    n_obs: int,
    observables: list[str],
    results_dir: str,
    init_prep_blochs: list | None = None,
) -> tuple[object, np.ndarray, float | None, float | None]:
    """Run multi-round SNPE training (plain NLL).

    Returns:
        (posterior, posterior_samples_np, nmae_mean, nmae_median)
        nmae_mean / nmae_median are None when theta_truth is None.
    """
    device = config["DEVICE"]
    n_rounds = config["SNPE_NUM_ROUNDS"]
    n_sims = config["SNPE_SIMS_PER_ROUND"]
    n_sims_first = int(config.get("SNPE_SIMS_FIRST_ROUND", n_sims))
    n_sims_other = int(n_sims)

    prior = BoxUniform(
        low=torch.zeros(num_params, device=device),
        high=torch.ones(num_params, device=device) * config["PRIOR_HIGH"],
    )
    prior_high = float(config["PRIOR_HIGH"])
    # Upper bound for inference samples: relaxed to allow posterior mass near the boundary.
    # Proposal sampling always uses strict [0, prior_high].
    infer_high = prior_high * float(config.get("INFER_UPPER_FACTOR", 1.0))
    x_for_posterior = x_obs.unsqueeze(0) if x_obs.dim() == 1 else x_obs  # updated per-round when STANDARDIZE_X

    train_kwargs = dict(
        training_batch_size=config["BATCH_SIZE"],
        learning_rate=config["LR"],
        max_num_epochs=config["EPOCHS"],
        stop_after_epochs=config["STOP_AFTER_EPOCHS"],
        show_train_summary=True,
        clip_max_norm=config["CLIP_MAX_NORM"],
    )

    def _make_npe():
        return NPE(prior=prior, density_estimator=_build_density_estimator(config, n_obs), device=device)

    # Print density estimator size
    _de_tmp = _build_density_estimator(config, n_obs)(
        torch.zeros(2, num_params, device=device), torch.zeros(2, n_obs, device=device)
    )
    print(f"  Density estimator parameters: {sum(p.numel() for p in _de_tmp.parameters()):,}")
    del _de_tmp

    _sims_desc = f"first={n_sims_first}, then={n_sims_other}"
    print(f"\nInitialising SNPE (rounds={n_rounds}, sims/round={_sims_desc}, "
          f"hidden={config['FLOW_HIDDEN_FEATURES']}, transforms={config['NUM_TRANSFORMS']})...")

    # ── stats tracking ────────────────────────────────────────────────────────
    mae_mean_hist:   list[float] = []
    mae_median_hist: list[float] = []
    n_total_hist:    list[int]   = []
    stats_path = os.path.join(results_dir, "posterior_mean_median_per_round.txt")
    r2_path    = os.path.join(results_dir, "r2_per_round.txt")

    truth_1d: np.ndarray | None = None
    if theta_truth is not None:
        truth_1d = theta_truth.detach().cpu().numpy().flatten()

    # ── Round 0: prior baseline ───────────────────────────────────────────────
    # Use the same sample count as the per-round posterior evaluation so that
    # r2_per_round.txt has a consistent n_samples column.
    n_eval_samples = int(config.get("NUM_POSTERIOR_SAMPLES", 2000))
    prior_samples = prior.sample((n_eval_samples,)).detach().cpu().numpy()
    prior_mean    = prior_samples.mean(axis=0)
    prior_median  = np.median(prior_samples, axis=0)
    with open(stats_path, "w") as f:
        f.write("# Round 0 (prior): mean and median\n")
        f.write("round,0\n")
        f.write(f"mean,{','.join(map(str, prior_mean))}\n")
        f.write(f"median,{','.join(map(str, prior_median))}\n")
    print(f"  Posterior mean/median per round → {stats_path}")
    # R^2 log: per-round R^2 for the scatter's 'All' and 'No gauge' panels.
    # n_samples column records how many posterior samples were used (consistency check).
    with open(r2_path, "w") as f:
        f.write("# Per-round R^2 of posterior mean vs truth (same sample-count as final scatter)\n")
        f.write("# columns: round,n_train,n_samples,r2_all,r2_no_gauge\n")
    if truth_1d is not None:
        ph = float(config["PRIOR_HIGH"])
        mae_mean_hist.append(float(np.mean(np.abs(prior_mean   - truth_1d)) / ph * 100))
        mae_median_hist.append(float(np.mean(np.abs(prior_median - truth_1d)) / ph * 100))
        n_total_hist.append(0)
        print(f"  Round 0 (prior) nMAE — mean: {mae_mean_hist[-1]:.2f}%,  median: {mae_median_hist[-1]:.2f}%")
        _r2_all_0      = _r2_mean_vs_truth(prior_samples, truth_1d)
        _keep_0        = _nogauge_keep_mask(prior_samples)
        _r2_nogauge_0  = _r2_mean_vs_truth(prior_samples[:, _keep_0], truth_1d[_keep_0])
        with open(r2_path, "a") as f:
            f.write(f"0,0,{prior_samples.shape[0]},{_r2_all_0:.6f},{_r2_nogauge_0:.6f}\n")
        print(f"  Round 0 (prior) R^2 — all: {_r2_all_0:.3f},  no_gauge: {_r2_nogauge_0:.3f}")

    # ── Initialise ────────────────────────────────────────────────────────────
    proposal  = prior
    posterior = None
    all_theta: list = []
    all_x:     list = []
    inference = None
    estimator = None

    n_jobs = config.get("SNPE_SIM_N_JOBS", 1)
    cx_topology = config["CX_TOPOLOGY"]
    n_qubits = config["N_QUBITS"]
    standardize_x = config.get("STANDARDIZE_X", True)
    # sims per round = number of simulator runs / (θ, x) rows per round. x is already the
    # full observation vector (all initial-state blocks × Pauli expectations); count once per row.
    def _n_theta_for_round(rnd_idx: int) -> int:
        return max(1, n_sims_first if rnd_idx == 0 else n_sims_other)

    n_total_so_far = 0
    final_samples_np: np.ndarray | None = None

    # Standardization stats (computed from round-1 data, reused for all rounds and x_obs)
    x_mean_np: np.ndarray | None = None
    x_std_np:  np.ndarray | None = None

    # ── Round loop ─────────────────────────────────────────────────────────────
    for rnd in range(n_rounds):
        n_theta = _n_theta_for_round(rnd)
        n_total_so_far += n_theta
        print(f"\n── SNPE round {rnd+1}/{n_rounds}  "
              f"({n_theta} simulations, obs_dim={n_obs}, cumulative={n_total_so_far}) ──")

        # Sample proposal (posterior or prior); reject out-of-support theta
        _is_post = hasattr(proposal, "potential_fn")

        def _sample(n):
            if _is_post:
                return proposal.sample((n,), show_progress_bars=False).detach().cpu().numpy()
            return proposal.sample((n,)).detach().cpu().numpy()

        accepted: list = []
        while sum(len(a) for a in accepted) < n_theta:
            batch = _sample(n_theta)
            batch = np.atleast_2d(batch)
            valid = batch[((batch >= 0).all(axis=1)) & ((batch <= prior_high).all(axis=1))]
            if len(valid):
                accepted.append(valid)
        theta_rnd_np = np.concatenate(accepted)[:n_theta]
        _theta_path = os.path.join(results_dir, f"theta_train_round{rnd+1}.npy")
        np.save(_theta_path, theta_rnd_np)
        print(f"  Saved theta samples → {_theta_path} (shape: {theta_rnd_np.shape})")

        print(f"  Running simulator ({n_theta} rows, obs_dim={n_obs})"
              + (f", {n_jobs} workers" if n_jobs > 1 else " (serial)") + "...")
        x_rnd_np = run_parallel(
            theta_rnd_np, n_qubits, observables,
            cx_topology, n_jobs, desc="Simulating",
            init_prep_blochs=init_prep_blochs,
        )

        _x_path = os.path.join(results_dir, f"x_train_round{rnd+1}.npy")
        np.save(_x_path, x_rnd_np)
        print(f"  Saved x samples    → {_x_path} (shape: {x_rnd_np.shape})")

        # ── x standardization (fit on round-1 data, reuse thereafter) ─────────
        if standardize_x:
            if rnd == 0:
                x_mean_np = x_rnd_np.mean(axis=0, keepdims=True)
                x_std_np  = x_rnd_np.std(axis=0,  keepdims=True).clip(1e-4)
                print(f"  Standardizing x: mean∈[{x_mean_np.min():.4f},{x_mean_np.max():.4f}], "
                      f"std∈[{x_std_np.min():.2e},{x_std_np.max():.2e}]  "
                      f"(n_near_const={(x_std_np < 1e-3).sum()})")
            x_rnd_np = np.clip((x_rnd_np - x_mean_np) / x_std_np, -10.0, 10.0)

        theta_rnd_t = torch.as_tensor(theta_rnd_np, dtype=torch.float32).to(device)
        x_rnd_t     = torch.as_tensor(x_rnd_np,    dtype=torch.float32).to(device)

        # ── Plain NLL training ────────────────────────────────────────────────
        all_theta.append(theta_rnd_np)
        all_x.append(x_rnd_np)
        theta_all = torch.as_tensor(np.concatenate(all_theta), dtype=torch.float32).to(device)
        x_all     = torch.as_tensor(np.concatenate(all_x),     dtype=torch.float32).to(device)
        if config.get("SNPE_REFRESH_MODEL", True) or rnd == 0:
            inference = _make_npe()
            inference.append_simulations(theta_all, x_all)
            label = f"fresh NLL ({len(theta_all)} samples)"
        else:
            inference.append_simulations(theta_rnd_t, x_rnd_t, proposal=prior)
            train_kwargs["force_first_round_loss"] = True
            label = f"warm-start NLL ({len(theta_all)} samples)"
        print(f"  Training round {rnd+1} [{label}]...")
        estimator = inference.train(**train_kwargs)
        plot_training_loss(inference, f"SNPE round {rnd+1}", results_dir, save_plot=False)

        # Apply same standardization to x_obs before building / querying posterior
        x_obs_for_post = x_obs
        if standardize_x and x_mean_np is not None:
            _xm = torch.as_tensor(x_mean_np, dtype=torch.float32).to(device)
            _xs = torch.as_tensor(x_std_np,  dtype=torch.float32).to(device)
            x_obs_for_post = ((x_obs - _xm) / _xs).clamp(-10.0, 10.0)
        x_for_posterior = x_obs_for_post.unsqueeze(0) if x_obs_for_post.dim() == 1 else x_obs_for_post

        posterior = _build_posterior_for_estimator(
            inference, estimator, prior, device, x_for_posterior, config
        )
        # Disable DirectPosterior's internal leakage-correction rejection loop;
        # we apply our own double-sided rejection below.
        try:
            posterior.prior = PermissivePrior()
        except AttributeError:
            try:
                posterior._prior = PermissivePrior()
            except AttributeError:
                pass
        proposal = posterior

        # ── Per-round MAE ─────────────────────────────────────────────────────
        # Cumulative (θ, x) training rows — used for scaling plots.
        n_total_hist.append(n_total_so_far)
        _eval_raw = posterior.sample(
            (1000,), x=x_for_posterior, show_progress_bars=False
        ).detach().cpu().numpy()
        _eval_raw = _ensure_samples_2d(_eval_raw, num_params)
        _eval_mask = ((_eval_raw >= 0) & (_eval_raw <= infer_high)).all(axis=1)
        eval_samples = _eval_raw[_eval_mask]
        if len(eval_samples) == 0:
            eval_samples = _eval_raw  # fallback: use unfiltered if all rejected
        eval_mean   = eval_samples.mean(axis=0)
        eval_median = np.median(eval_samples, axis=0)
        with open(stats_path, "a") as f:
            f.write(f"round,{rnd+1}\n")
            f.write(f"mean,{','.join(map(str, eval_mean))}\n")
            f.write(f"median,{','.join(map(str, eval_median))}\n")
        if truth_1d is not None:
            _nm = float(np.mean(np.abs(eval_mean - truth_1d)) / prior_high * 100)
            _nd = float(np.mean(np.abs(eval_median - truth_1d)) / prior_high * 100)
            mae_mean_hist.append(_nm)
            mae_median_hist.append(_nd)
            print(f"  Round {rnd+1} nMAE — mean: {_nm:.2f}%,  median: {_nd:.2f}%")

        # ── Per-round posterior samples ───────────────────────────────────────
        _n_post = config["NUM_POSTERIOR_SAMPLES"]
        _rnd_list, _collected = [], 0
        _batch_sz = max(200, _n_post // 5)
        for _ in range(500):
            if _collected >= _n_post:
                break
            b = posterior.sample((_batch_sz,), x=x_for_posterior,
                                 show_progress_bars=False).detach().cpu().numpy()
            b = _ensure_samples_2d(b, num_params)
            v = b[((b >= 0).all(axis=1)) & ((b <= infer_high).all(axis=1))]
            if len(v):
                _rnd_list.append(v); _collected += len(v)
        if _rnd_list:
            _rnd_np = np.concatenate(_rnd_list)[:_n_post]
        else:
            _rnd_np = np.empty((0, num_params))
        if len(_rnd_np) < _n_post:
            print(f"  Warning: Only {len(_rnd_np)}/{_n_post} in-support posterior samples (0..{infer_high}).")
        if rnd + 1 == n_rounds:
            final_samples_np = _rnd_np.copy()

        # ── Per-round R^2 (matches scatter's 'All' and 'No gauge' panels) ─────
        if truth_1d is not None and len(_rnd_np) > 0:
            _r2_all     = _r2_mean_vs_truth(_rnd_np, truth_1d)
            _keep       = _nogauge_keep_mask(_rnd_np)
            _r2_nogauge = _r2_mean_vs_truth(_rnd_np[:, _keep], truth_1d[_keep])
            with open(r2_path, "a") as f:
                # n_train = cumulative (θ, x) training rows up to this round.
                f.write(f"{rnd+1},{n_total_so_far},{_rnd_np.shape[0]},"
                        f"{_r2_all:.6f},{_r2_nogauge:.6f}\n")
            print(f"  Round {rnd+1} R^2 — all: {_r2_all:.3f},  no_gauge: {_r2_nogauge:.3f}")

    # ── Save final model ──────────────────────────────────────────────────────
    if estimator is not None:
        model_path = os.path.join(results_dir, "estimator.pt")
        torch.save(estimator, model_path)
        print(f"  Model saved → {model_path}")

    nmae_mean   = mae_mean_hist[-1]   if mae_mean_hist   else None
    nmae_median = mae_median_hist[-1] if mae_median_hist else None

    # Attach history for scaling plot
    posterior._snpe_mae_mean      = mae_mean_hist
    posterior._snpe_mae_median    = mae_median_hist
    posterior._snpe_n_total       = n_total_hist
    posterior._snpe_sims_per_round = n_sims_other

    # Persist the per-round totals for reproduce()/scaling plots.
    n_total_path = os.path.join(results_dir, "snpe_n_total_per_round.txt")
    with open(n_total_path, "w") as f:
        f.write("# columns: round (1-indexed), n_sims_this_round, n_total_cumulative\n")
        n_tot = 0
        for i in range(n_rounds):
            n_i = _n_theta_for_round(i)
            n_tot += n_i
            f.write(f"{i+1},{n_i},{n_tot}\n")

    return posterior, final_samples_np, nmae_mean, nmae_median
