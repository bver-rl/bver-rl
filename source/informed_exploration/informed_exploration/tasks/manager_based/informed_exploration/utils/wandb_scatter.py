"""Periodic xy scatter of curriculum buffer contents, logged to wandb as one small image; Isaac-free.

Logged with ``commit=False`` so the image merges into the writer's next row; logging with its own step
would make wandb drop subsequent scalar rows.
"""

from __future__ import annotations

import torch

# CVD-safe blue/orange pair; the subplot title identifies each series, so no legend is needed.
GOAL_COLOR = "#ff7f0e"
START_COLOR = "#1f77b4"
_BOUNDS_COLOR = "#bbbbbb"
_ANCHOR_COLOR = "#222222"

_warned_no_wandb = False


def wandb_available() -> bool:
    """Return whether a wandb run is active in this process, warning once otherwise."""
    global _warned_no_wandb
    try:
        import wandb

        run = wandb.run
    except ImportError:
        run = None
    if run is None:
        if not _warned_no_wandb:
            _warned_no_wandb = True
            print("[WARN] wandb_scatter: no active wandb run; buffer scatter logging disabled")
        return False
    return True


def subsample_xy(collection, xy_dims: tuple[int, int], num_points: int):
    """Draw at most ``num_points`` xy points uniformly without replacement from a collection.

    Subsamples before stacking so only the drawn elements are copied to CPU.

    Args:
        collection: Sequence of same-shape tensors, flattened before indexing.
        xy_dims: Indices of x and y into the flattened task vector.
        num_points: Maximum number of points drawn.

    Returns:
        An ``[n, 2]`` float numpy array, or ``None`` for an empty collection.
    """
    if len(collection) == 0:
        return None
    items = list(collection)
    if len(items) > num_points:
        keep = torch.randperm(len(items))[:num_points]
        items = [items[i] for i in keep.tolist()]
    flat = torch.stack(items).reshape(len(items), -1)
    return flat[:, list(xy_dims)].detach().cpu().numpy()


def log_mixed_buffer_scatter(
    key: str,
    goal_xy,
    start_xys: list,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    anchor_xys: list[tuple[float, float]] | None = None,
) -> None:
    """Render the goal scatter and per-anchor start scatters as one figure and log it under ``key``.

    Limits are fixed to ``xy_bounds`` so consecutive images stay comparable. The caller must check
    :func:`wandb_available` first.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb
    from matplotlib.patches import Rectangle

    n_plots = 1 + len(start_xys)
    ncols = min(3, n_plots)
    nrows = -(-n_plots // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), dpi=100, squeeze=False)
    flat_axes = axes.ravel()

    series = [("goals", goal_xy, GOAL_COLOR, None)]
    for k, xy in enumerate(start_xys):
        anchor = anchor_xys[k] if anchor_xys is not None and k < len(anchor_xys) else None
        series.append((f"starts a{k}", xy, START_COLOR, anchor))

    for ax, (title, xy, color, anchor) in zip(flat_axes, series):
        if xy is not None and len(xy) > 0:
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c=color, alpha=0.6, linewidths=0)
        if anchor is not None:
            ax.plot(anchor[0], anchor[1], marker="x", color=_ANCHOR_COLOR, markersize=6, markeredgewidth=1.5)
        if xy_bounds is not None:
            (x_lo, x_hi), (y_lo, y_hi) = xy_bounds
            ax.add_patch(
                Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, edgecolor=_BOUNDS_COLOR, linewidth=0.8)
            )
            pad_x = 0.05 * (x_hi - x_lo) or 0.5
            pad_y = 0.05 * (y_hi - y_lo) or 0.5
            ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
            ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8, color="#444444")
        ax.tick_params(labelsize=6, length=2, colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.5)
    wandb.log({key: wandb.Image(fig)}, commit=False)
    plt.close(fig)
