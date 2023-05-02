# %% Imports
from mne.preprocessing import (
    compute_bridged_electrodes,
    interpolate_bridged_electrodes,
)
from mne.viz import plot_bridged_electrodes

from eeg_cybersickness.io import read_raw
from eeg_cybersickness.utils.path import get_derivative_stem


# %% Load raw recording
root = "/mnt/Isilon/9003_CBT_HNP_MEEG/projects/project_cybersickness/data"
participant = 57
session = 1
raw = read_raw(root, participant, session)
derivative_stem = get_derivative_stem(root, participant, session)

# %% Fix gel-bridges
raw.set_montage("standard_1020")
bridged_idx, ed_matrix = compute_bridged_electrodes(raw)
_ = plot_bridged_electrodes(
    raw.info,
    bridged_idx,
    ed_matrix,
    title="Bridged Electrodes",
    topomap_args=dict(vlim=(0, 5)),
)
interpolate_bridged_electrodes(raw, bridged_idx)
raw.set_montage(None)

# %% Filter and annotate bad channels/segments
raw.filter(
    l_freq=1.0,
    h_freq=30.0,
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
raw.plot(theme="light")

# %% Save pre-ICA raw recording
fname = derivative_stem.with_name(f"{derivative_stem.name}-raw.fif")
raw.save(fname)
