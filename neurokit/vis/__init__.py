import numpy as np
import plotly.graph_objs as go


def plot_spectrogram(f, t, S, fmin=0, fmax=40, autoclip=True, scale='db'):
    scale = scale.lower() if scale is not None else 'linear'
    if scale not in ['db', 'linear']:
        raise ValueError('Invalid scale: allowed values are `db` or `linear`')

    S = S[(f >= fmin) & (f <= fmax)]
    freqs = f[(f >= fmin) & (f <= fmax)]

    if autoclip:
        zmin = np.nanquantile(S, 0.01)
        zmax = np.nanquantile(S, 0.99)
        S = S.clip(zmin, zmax)

    if scale.lower() == 'db':
        S = np.log10(S)

    return go.Figure(data=go.Heatmap(x=t, y=freqs, z=S,
                                     colorscale='spectral',
                                     reversescale=True, showscale=False))
