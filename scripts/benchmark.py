"""Compares artifact and suppressions detection with a reference.

The script takes a single positional argument `data_path` which is the path to
a folder containing a `masks.csv` file with the reference masks and the raw
data file `data.edf` in EDF format, containing the original recording.
The number of samples in the data must correspond to the length of the masks.

Usage: python benchmark.py path/to/my/data_folder
"""
import argparse
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import neurokit as nk
import pandas as pd
import plotly.io

from neurokit.preprocessing import (HighAmplitudeDetector,
                                    ConstantSignalDetector)
from neurokit.analysis.suppressions import SuppressionAnalyzer

plotly.io.templates.default = 'plotly_white'

# %%


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_path', metavar='data_path', type=str,
                    help='Path to the data directory to use, e.g. `./data/artifacts`')

args = parser.parse_args()
data_path = Path(args.data_path)

# %%

# Read EDF file
rec = nk.io.read_edf(data_path.joinpath('data.edf'))
_detectors = [HighAmplitudeDetector(), ConstantSignalDetector(interval=32)]
rec.artifacts = nk.preprocessing.detect_artifacts(rec, detectors=_detectors)

# Read reference masks and transform to intervals
masks = pd.read_csv(data_path.joinpath('masks.csv'), dtype=bool)

ref_artifacts = nk.utils.mask_to_intervals(masks.artefact, rec.data.index)
ref_isoel_sup = nk.utils.mask_to_intervals(masks['ies'], rec.data.index)
ref_alpha_sup = nk.utils.mask_to_intervals(masks['as'], rec.data.index)

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.025)

# Plot the main signal
for n, ch in enumerate(rec.channels):
    trace = go.Scattergl(x=rec.data.index, y=rec.data[ch],
                         line_width=1, mode='lines', line_color='#56606B')
    fig.add_trace(trace, row=n + 1, col=1)


# Plot the reference
# ==================

# Artifacts
for start, end in ref_artifacts:
    for n, ch in enumerate(rec.channels):
        _data = rec.data.loc[start:end]
        trace = go.Scattergl(x=_data.index, y=_data[ch],
                             line_width=1.25, mode='lines',
                             line_color='#AE2222')
        fig.add_trace(trace, row=n + 1, col=1)

# Iso-electric suppressions
for start, end in ref_isoel_sup:
    for n, ch in enumerate(rec.channels):
        _data = rec.data.loc[start:end]
        trace = go.Scattergl(x=_data.index, y=_data[ch],
                             line_width=1.25, mode='lines',
                             line_color='#6236FF')
        fig.add_trace(trace, row=n + 1, col=1)

for start, end in ref_alpha_sup:
    for n, ch in enumerate(rec.channels):
        _data = rec.data.loc[start:end]
        trace = go.Scattergl(x=_data.index, y=_data[ch],
                             line_width=1.25, mode='lines',
                             line_color='#F7B500')
        fig.add_trace(trace, row=n + 1, col=1)

# Plot the detections
# ===================
pad = 0.01

for artifact in rec.artifacts.itertuples():
    n_ch = rec.channels.index(artifact.channel)
    shape = {'type': 'rect', 'xref': 'x', 'yref': 'paper', 'layer': 'below',
             'x0': artifact.start, 'x1': artifact.end,
             'line_width': 0, 'fillcolor': '#AE2222', 'opacity': 0.5,
             'y0': 1 - 0.25 * n_ch - pad, 'y1': 1 - 0.25 * (n_ch + 1) + pad}
    fig.add_shape(shape)

analyzer = SuppressionAnalyzer(rec)
isoel_sup = analyzer.detect_ies()
alpha_sup = analyzer.detect_alpha_suppressions()
for sup in alpha_sup.itertuples():
    for n in range(4):
        shape = {'type': 'rect', 'xref': 'x', 'yref': 'paper', 'layer': 'below',
                 'x0': sup.start, 'x1': sup.end,
                 'line_width': 0, 'fillcolor': '#F7B500', 'opacity': 0.5,
                 'y0': 0.25 * n + 2 * pad, 'y1': 0.25 * (n + 1) - 2 * pad}
        fig.add_shape(shape)


for sup in isoel_sup.itertuples():
    for n in range(4):
        shape = {'type': 'rect', 'xref': 'x', 'yref': 'paper', 'layer': 'below',
                 'x0': sup.start, 'x1': sup.end,
                 'line_width': 0, 'fillcolor': '#6236FF', 'opacity': 0.5,
                 'y0': 0.25 * n + 3 * pad, 'y1': 0.25 * (n + 1) - 3 * pad}
        fig.add_shape(shape)


fig.update_layout(showlegend=False,
                  margin=dict(autoexpand=True, l=5, r=5, t=5, b=5, pad=0))
fig.update_layout(yaxis_title=rec.channels[0],
                  yaxis2_title=rec.channels[1],
                  yaxis3_title=rec.channels[2],
                  yaxis4_title=rec.channels[3])
fig.show(renderer='browser')
