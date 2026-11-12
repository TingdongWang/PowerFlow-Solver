"""Heatmap of worst per-sample PF mismatch in each perturbation cell."""
import numpy as np
import pandas as pd


def cell_maxima(frame):
    frame = frame.copy()
    for key in ('x_signed', 'pq_signed'):
        if not np.isfinite(frame[key]).all():
            raise ValueError('Nonfinite heatmap coordinates')
    for axis, key in [('x','x_signed'), ('pq','pq_signed')]:
        if axis+'_low' in frame and axis+'_high' in frame:
            frame[key] = ((frame[axis+'_low']+frame[axis+'_high'])/2).round(6)
        else:
            # Legacy CSV has sampled coordinates, not cell centers.
            edges = np.linspace(-2., 2., 21)
            if ((frame[key] < -2) | (frame[key] > 2)).any():
                raise ValueError('Legacy CSV coordinates outside the known [-2,2] grid')
            indices = np.minimum(np.searchsorted(edges, frame[key], side='right')-1, 19)
            frame[key] = ((edges[indices]+edges[indices+1])/2).round(6)
    if (frame['max_pf_mismatch'] < 0).any():
        raise ValueError('PF mismatch magnitude cannot be negative')
    # A failed nonfinite sample must not disappear through pandas skipna.
    frame['max_pf_mismatch'] = frame['max_pf_mismatch'].where(
        np.isfinite(frame['max_pf_mismatch']), np.inf)
    grouped = frame.groupby(['pq_signed', 'x_signed'])['max_pf_mismatch']
    worst = grouped.max().unstack().sort_index().sort_index(axis=1)
    counts = grouped.size().unstack().reindex(index=worst.index, columns=worst.columns)
    if counts.isna().any().any():
        raise ValueError('Missing heatmap cells; refusing to hide incomplete grid')
    return worst, counts


def cell_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0]-.1, centers[0]+.1])
    middle = (centers[:-1]+centers[1:])/2
    return np.r_[centers[0]-(centers[1]-centers[0])/2,
                 middle, centers[-1]+(centers[-1]-centers[-2])/2]


def save_heatmap(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    frame = pd.read_csv(out/'heatmap.csv')
    worst, counts = cell_maxima(frame)
    values = worst.to_numpy()
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    # Match the reference scale, expanding it if actual values exceed it.
    vmin = min(1e-5, float(positive.min())) if positive.size else 1e-5
    vmax = max(2e-3, float(positive.max())) if positive.size else 2e-3
    display = np.ma.array(np.maximum(values, vmin), mask=~finite)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#00ffff')
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    mesh = ax.pcolormesh(cell_edges(worst.columns), cell_edges(worst.index), display,
                         norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, shading='flat')
    ax.set_xticks(worst.columns.to_numpy()[::2])
    ax.set_yticks(worst.index.to_numpy()[::2])
    for boundary in (-1., 1.):
        if ax.get_xlim()[0] < boundary < ax.get_xlim()[1]:
            ax.axvline(boundary, color='white', lw=.7, alpha=.8)
        if ax.get_ylim()[0] < boundary < ax.get_ylim()[1]:
            ax.axhline(boundary, color='white', lw=.7, alpha=.8)
    ax.set(xlabel='Initial state perturbation X', ylabel='Specified P/Q perturbation',
           title=f'{out.name} | Worst residual · independent sample criterion')
    fig.colorbar(mesh, ax=ax, pad=.04, label='Worst sample max PF mismatch in cell (p.u.)')
    failures = int((~finite).sum())
    if failures:
        ax.text(.01, .01, f'Cyan: {failures} cells contain nonfinite residuals',
                transform=ax.transAxes, color='black', bbox={'facecolor':'white','alpha':.9})
    fig.tight_layout()
    fig.savefig(out/'heatmap.png', dpi=180)
    plt.close(fig)
    return out/'heatmap.png'
