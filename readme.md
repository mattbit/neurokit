# Neurokit

[![Build Status](https://travis-ci.com/mattbit/neurokit.svg?token=zKpBnjBx4d1NEMb7zFbd&branch=master)](https://travis-ci.com/mattbit/neurokit)

A basic toolbox to deal with neurophysiological timeseries.

---
## Installation

Package is not public, but it can be installed from a local folder for development:

```sh
$ git clone git@github.com:mattbit/neurokit.git /path/to/neurokit
$ pip install -e /path/to/neurokit
```

or via ssh with pip:
```sh
$ pip install git+ssh://git@github.com:mattbit/neurokit.git
```

## Getting started

```python
import neurokit as nk

# Read an EDF file
recording = nk.io.read_edf('my_file.edf')

# The pandas dataframe with all signals
recording.data.head()

# The annotations
recording.annotations.head()

# Calculate a spectrogram
freq, time, S = nk.signal.spectrogram(recording, channels=['O1', 'O2'], window=1, overlap=0.75)

# and plot
fig = nk.vis.plot_spectrogram(freq, time, S)
fig.show()
```


## Fixing and anonymizing EDF files (WIP!)

```python
import neurokit as nk

nk.io.edf.fix_edf('/file/to/fix.edf', '/the/fixed/output.edf', anonymize=True)
```
