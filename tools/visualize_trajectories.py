"""Visualize FM / IMLE sampled trajectories against ground truth.

Loads sample pkl files saved by `trainer/denoising_model_trainers.py` and
`trainer/imle_trainers.py`, and renders per-scene PNGs overlaying:
  - observed past (black)
  - ground-truth future (green)
  - K predicted futures (light, per-agent hue)
"""
import argparse
import os
import pickle
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

REQUIRED_KEYS = ("past_traj_original_scale", "fut_traj_original_scale", "y_pred_data")


def load_samples_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return data


def iter_pkl_files(samples_dir, samples_pkl):
    if samples_pkl:
        return [Path(samples_pkl)]
    if not samples_dir:
        raise ValueError("Either --samples_dir or --samples_pkl must be given")
    files = sorted(glob(os.path.join(samples_dir, "*_batch_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No '*_batch_*.pkl' files in {samples_dir}")
    return [Path(p) for p in files]


def _connect(past_xy, fut_xy):
    """Prepend the last past point to the future segment so lines join visually."""
    return np.concatenate([past_xy[-1:], fut_xy], axis=0)


def plot_scene(past, gt, preds, save_path, title, num_preds=None):
    """
    past:  [A, T_past, 2]
    gt:    [A, T_fut,  2]
    preds: [K, A, T_fut, 2]
    """
    A = past.shape[0]
    K = preds.shape[0]
    if num_preds is not None and num_preds < K:
        idx = np.linspace(0, K - 1, num_preds).astype(int)
        preds = preds[idx]
        K = preds.shape[0]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#f4f1ea")
    ax.set_facecolor("#f4f1ea")
    cmap = plt.get_cmap("tab10")

    # white outline so foreground lines stay readable over the prediction cloud
    halo_white = [pe.withStroke(linewidth=4.5, foreground="white")]
    halo_white_thin = [pe.withStroke(linewidth=3.0, foreground="white")]

    # --- Layer 1 (background): all predictions ---
    for a in range(A):
        agent_color = cmap(a % 10)
        for k in range(K):
            pr_line = _connect(past[a], preds[k, a])
            label = "prediction" if (a == 0 and k == 0) else None
            ax.plot(
                pr_line[:, 0], pr_line[:, 1],
                color=agent_color, alpha=0.28, linewidth=1.2,
                solid_capstyle="round", zorder=2, label=label,
            )
            ax.scatter(
                preds[k, a, -1, 0], preds[k, a, -1, 1],
                color=agent_color, alpha=0.35, s=14, linewidths=0, zorder=2.1,
            )

    # --- Layer 2: past trajectory with white halo ---
    for a in range(A):
        label = "past" if a == 0 else None
        ax.plot(
            past[a, :, 0], past[a, :, 1],
            color="#222222", linewidth=2.2, solid_capstyle="round",
            path_effects=halo_white, zorder=3, label=label,
        )
        ax.scatter(
            past[a, 0, 0], past[a, 0, 1],
            color="#222222", s=40, marker="o",
            edgecolors="white", linewidths=1.2, zorder=3.1,
        )

    # --- Layer 3 (foreground): GT future with strong halo ---
    gt_color = "#d62728"
    for a in range(A):
        label = "GT future" if a == 0 else None
        gt_line = _connect(past[a], gt[a])
        ax.plot(
            gt_line[:, 0], gt_line[:, 1],
            color=gt_color, linewidth=2.6, solid_capstyle="round",
            path_effects=halo_white, zorder=4, label=label,
        )
        ax.scatter(
            gt[a, -1, 0], gt[a, -1, 1],
            color=gt_color, s=130, marker="*",
            edgecolors="white", linewidths=1.4, zorder=4.1,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=10)
    # map-like: hide axis chrome, soft grid
    ax.tick_params(left=False, right=False, top=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="white", linewidth=1.2, alpha=0.9, zorder=1)
    ax.margins(0.08)

    leg = ax.legend(loc="best", fontsize=8, frameon=True,
                    facecolor="white", edgecolor="#cccccc")
    leg.set_zorder(5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _resolve_output_dir(pkl_path, samples_dir, output_dir):
    if output_dir:
        base = Path(output_dir)
    else:
        anchor = Path(samples_dir) if samples_dir else pkl_path.parent
        base = anchor.parent / "viz"
    return base / pkl_path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=str, default=None)
    parser.add_argument("--samples_pkl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=8)
    parser.add_argument("--scene_indices", type=str, default=None,
                        help="Comma-separated scene indices to render (overrides --max_scenes)")
    parser.add_argument("--num_preds", type=int, default=None,
                        help="Subsample at most this many prediction samples per agent")
    args = parser.parse_args()

    pkl_files = iter_pkl_files(args.samples_dir, args.samples_pkl)
    scene_indices = None
    if args.scene_indices:
        scene_indices = [int(s) for s in args.scene_indices.split(",") if s.strip()]

    for pkl_path in pkl_files:
        print(f"[load] {pkl_path}")
        data = load_samples_pkl(pkl_path)
        past_all = data["past_traj_original_scale"]   # [B, A, Tp, F]  (abs|rel|vel)
        gt_all = data["fut_traj_original_scale"]      # [B, A, Tf, 2]  (relative to past end)
        pred_all = data["y_pred_data"]                # [B, K, A, Tf, 2] (relative to past end)

        # past[...,:2] is absolute world coords; fut / predictions are stored
        # relative to the last past frame (see dataloader_eth_ucy.py:127).
        # Lift fut/preds into world coords so trajectories connect cleanly.
        past_all = past_all[..., :2]
        anchor = past_all[:, :, -1:, :]                                  # [B, A, 1, 2]
        gt_all = gt_all[..., :2] + anchor                                # [B, A, Tf, 2]
        pred_all = pred_all[..., :2] + anchor[:, None, :, :, :]          # [B, K, A, Tf, 2]

        B = past_all.shape[0]
        if scene_indices is not None:
            indices = [b for b in scene_indices if 0 <= b < B]
        else:
            indices = list(range(min(B, args.max_scenes)))

        out_dir = _resolve_output_dir(pkl_path, args.samples_dir, args.output_dir)
        for b in indices:
            past = past_all[b]
            gt = gt_all[b]
            preds = pred_all[b]
            title = f"{pkl_path.stem}\nscene={b}  agents={past.shape[0]}  K={preds.shape[0]}"
            save_path = out_dir / f"scene_{b:04d}.png"
            plot_scene(past, gt, preds, save_path, title, num_preds=args.num_preds)
            print(f"  -> {save_path}")


if __name__ == "__main__":
    main()
