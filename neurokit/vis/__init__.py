import numpy as np
import plotly.graph_objs as go


def plot_spectrogram(f, t, S, fmin=0, fmax=40, autoclip=True):
    S = S[(f >= fmin) & (f <= fmax)]
    freqs = f[(f >= fmin) & (f <= fmax)]

    if autoclip:
        zmin = np.nanquantile(S, 0.01)
        zmax = np.nanquantile(S, 0.99)
        S = S.clip(zmin, zmax)

    return go.Figure(data=go.Heatmap(x=t, y=freqs, z=20 * np.log10(S),
                                     colorscale='spectral',
                                     reversescale=True, showscale=False))
