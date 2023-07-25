from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from mne import Epochs, make_fixed_length_events, pick_info
from mne.io import BaseRaw, read_raw_fif
from mne.io.pick import _picks_to_idx
from numpy.typing import NDArray
from scipy.integrate import simpson


def compute_bandpower(
    raw: BaseRaw, start: float, stop: float
) -> Tuple[Dict[str, NDArray[float]], int]:
    """Compute the relative bandpower on the raw segment.

    Parameters
    ----------
    raw : Raw
        Continuous recording.
    start : float
        Start of the window on which the bandpower is computed, in seconds.
    stop : float
        End of the window on which the bandpower is computed, in seconds.

    Returns
    -------
    bandpowers : dict
        The key is the name (str) of the band.
        The value is the average relative bandpower in that range
        (array of shape (n_good_channels,)).
    n_epochs : int
        Number of epochs used to compute the bandpower. The maximum number is 581.
    """
    bandpowers = dict()
    events = make_fixed_length_events(
        raw, start=start, stop=stop, duration=2, overlap=1.9, first_samp=True
    )
    epochs = Epochs(
        raw,
        events,
        tmin=0,
        tmax=2,
        baseline=None,
        picks="eeg",
        reject_by_annotation=True,
        preload=True,
    )
    reject = get_rejection_threshold(epochs)
    epochs.drop_bad(reject=reject)
    if len(epochs) == 0:
        return bandpowers, 0
    spectrum = epochs.compute_psd(
        method="welch",
        n_fft=int(2 * raw.info["sfreq"]),
        n_per_seg=int(2 * raw.info["sfreq"]),
        fmin=raw.info["highpass"],  # 1 or 4 Hz
        fmax=30.0,
    )
    freq_res = spectrum.freqs[1] - spectrum.freqs[0]
    psd_full = spectrum.get_data(fmin=raw.info["highpass"], fmax=30)
    bp_full = simpson(psd_full, dx=freq_res, axis=-1)
    for band, (fmin, fmax) in bands.items():
        if raw.info["highpass"] == 4 and band == "delta":
            continue
        psd = spectrum.get_data(fmin=fmin, fmax=fmax)
        bp = simpson(psd, dx=freq_res, axis=-1) / bp_full
        bandpowers[band] = np.average(bp, axis=0)
    return bandpowers, len(epochs)


root = Path("/mnt/Isilon/9003_CBT_HNP_MEEG/projects/project_cybersickness/data/")
session = 2
directory = root / f"derivatives-session-{session}"
participants = [9, 12, 23, 28, 31, 32, 34, 36, 57, 58]
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

for k, participant in enumerate(participants):
    participant_str = str(participant).zfill(2)
    fname = directory / f"P{participant_str}" / f"P{participant_str}_S{session}-raw.fif"
    del participant_str
    raw = read_raw_fif(fname, preload=True)
    if k == 0:
        eeg_ch_names = pick_info(
            raw.info, _picks_to_idx(raw.info, picks="eeg", exclude=())
        ).ch_names
        keys = ["participant", "times", "n_epochs"] + eeg_ch_names
        dfs = {band: dict() for band in bands}
        for band in bands:
            for key in keys:
                dfs[band][key] = list()

    for tmin in np.arange(0, raw.times[-1], 60):
        bandpowers, n_epochs = compute_bandpower(raw, tmin, tmin + 60)
        # fill dataframes
        for band in bands:
            dfs[band]["participant"].append(participant)
            dfs[band]["times"].append(tmin)
            dfs[band]["n_epochs"].append(n_epochs)
            if band in bandpowers:
                counter = 0
                for ch in eeg_ch_names:
                    if ch in raw.info["bads"]:
                        dfs[band][ch].append(np.nan)
                    else:
                        dfs[band][ch].append(bandpowers[band][counter])
                        counter += 1
            else:
                for ch in eeg_ch_names:
                    dfs[band][ch].append(np.nan)

    del raw

df_delta = pd.DataFrame.from_dict(dfs["delta"])
df_theta = pd.DataFrame.from_dict(dfs["theta"])
df_alpha = pd.DataFrame.from_dict(dfs["alpha"])
df_beta = pd.DataFrame.from_dict(dfs["beta"])
