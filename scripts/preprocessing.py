# %% Imports
from mne.preprocessing import (
    ICA,
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
plot_bridged_electrodes(
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
    h_freq=40.0,
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
raw.plot(theme="light")

# %% Save pre-ICA raw recording
fname = derivative_stem.with_name(f"{derivative_stem.name}-pre-ica-raw.fif")
raw.save(fname)

# %% Run ICA
ica = ICA(n_components=None, method="picard")
raw.set_montage("standard_1020")
ica.fit(raw)
ica.plot_components(inst=raw)
ica.plot_sources(inst=raw, theme="light")
fname = derivative_stem.with_name(f"{derivative_stem.name}-ica.fif")
ica.save(fname)
raw.set_montage(None)

# %% Apply ICA
ica.apply(raw)

# %% Add reference CPz
raw.add_reference_channels(raw, ref_channels="CPz", copy=False)
raw.set_eeg_reference("average", ch_type="eeg")
raw.set_montage("standard_1020")
fname = derivative_stem.with_name(f"{derivative_stem.name}-raw.fif")
raw.save(fname)
