[![PyPI license](https://img.shields.io/pypi/l/thunderfish.svg)](https://pypi.python.org/pypi/thunderfish/)
[![Build Status](https://travis-ci.org/bendalab/thunderfish.svg?branch=master)](https://travis-ci.org/bendalab/thunderfish)
[![codecov](https://codecov.io/gh/bendalab/thunderfish/branch/master/graph/badge.svg)](https://codecov.io/gh/bendalab/thunderfish)
[![PyPI version](https://badge.fury.io/py/thunderfish.svg)](https://badge.fury.io/py/thunderfish)

# ThunderFish

Algorithms and programs for analysing electric field recordings of
weakly electric fish.

[Documentation](https://bendalab.github.io/thunderfish) |
[API Reference](https://bendalab.github.io/thunderfish/api)

Weakly electric fish generate an electric organ discharge (EOD).  In
wave-type fish the EOD resembles a sinewave of a specific frequency
and with higher harmonics. In pulse-type fish EODs have a distinct
waveform and are separated in time. The thunderfish package provides
algorithms and tools for analysing both wavefish and pulsefish EODs.


## Installation

Simply run (as superuser):
```
pip install thunderfish
```

If you have problems loading specific audio files with thunderfish,
then you need to install further packages. Follow the [installation
instructions](https://bendalab.github.io/audioio/installation/) of the
[AudioIO](https://bendalab.github.io/audioio/) package.


## Software

The thunderfish package provides the following software:

- *fishfinder*: Browse EOD recordings and detect EOD frequencyies on the fly.
- *thunderfish*: Automatically detect and analyze all EOD waveforms in a short recording and generate a summary plot and data tables. [Read documentation](docs/thunderfish.md).
- *collectfish*: Collect data generated by thunderfish. [Read documentation](docs/collectfish.md).
- *eodexplorer*: View and explore properties of EOD waveforms. [Read documentation](docs/eodexplorer.md).
- *thunderlogger*: extract EOD waveforms from logger recordings.


## Algorithms

The following modules provide the algorithms for analyzing EOD recordings.
Look into the modules for more information.

### Input/output

- *configfile.py*: Configuration file with help texts for analysis parameter.
- *consoleinput.py*: User input from console.
- *dataloader.py*: Load time-series data from files.
- *tabledata.py*: Read and write tables with a rich hierarchical header including units and formats.

### Basic data analysis

- *eventdetection.py*: Detect and hande peaks and troughs as well as threshold crossings in data arrays.
- *powerspectrum.py*: Compute and plot powerspectra and spectrograms for a given minimum frequency resolution.
- *voronoi.py*: Analyse Voronoi diagrams based on scipy.spatial.
- *multivariateexplorer.py*: Simple GUI for viewing and exploring multivariate data.

### EOD analysis

- *bestwindow.py*: Select the region within a recording with the most stable signal of largest amplitude that is not clipped.
- *checkpulse.py*: Check whether a pulse-type or a wave-type weakly electric fish is present in a recording.
- *consistentfishes.py*: Create a list of EOD frequencies with fishes present in all provided fish lists.
- *eodanalysis.py*: Analyse EOD waveforms.
- *harmonics.py*: Extract and analyze harmonic frequencies from power spectra.
- *pulses.py*: Extract and cluster EOD waverforms of pulse-type electric fish.

### EOD simulations

- *fakefish.py*: Simulate EOD waveforms.
- *efield.py*: Simulations of spatial electric fields.
- *fishshapes.py*: Manipulate and plot fish outlines.



