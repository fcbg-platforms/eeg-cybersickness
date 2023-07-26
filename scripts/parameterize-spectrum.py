from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from fooof import FOOOF
from mne import Epochs, make_fixed_length_events, pick_info
from mne.io import BaseRaw, read_raw_fif
from mne.io.pick import _picks_to_idx
from numpy.typing import NDArray


def parameterize_spectrum(
    raw: BaseRaw, start: float, stop: float
) -> Tuple[NDArray[float], NDArray[float], NDArray[float], int]:
    """Parameterize the aperiodic component of the spectrum.

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
    r_squared : array of shape (n_good_channels,)
        R² of the fit between the input power spectrum and the full model fit,
        per channel.
    error : array of shape (n_good_channels,)
        Error of the full model fit, per channel.
    aperiodic_params : array of shape (n_good_channels, 2)
        Parameters that define the aperiodic fit, as (Offset, Exponent), per channel.
    n_epochs : int
        Number of epochs used to compute the bandpower. The maximum number is 581.
    """
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
        return None
    # array of shape (n_channels, n_freqs)
    spectrum = epochs.compute_psd(
        method="welch",
        n_fft=int(2 * raw.info["sfreq"]),
        n_per_seg=int(2 * raw.info["sfreq"]),
        fmin=raw.info["highpass"],  # 1 or 4 Hz
        fmax=30.0,
    )
    freqs = spectrum.freqs
    data = spectrum.get_data().mean(axis=0)
    r_squared = np.zeros(data.shape[0])
    error = np.zeros(data.shape[0])
    aperiodic_params = np.zeros((data.shape[0], 2))
    fm = FOOOF(peak_width_limits=(2, 12))
    for k, psd in enumerate(data):
        fm.fit(freqs, psd)
        r_squared[k] = fm.r_squared_
        error[k] = fm.error_
        aperiodic_params[k] = fm.aperiodic_params_
    return r_squared, error, aperiodic_params, len(epochs)


root = Path("/mnt/Isilon/9003_CBT_HNP_MEEG/projects/project_cybersickness/data/")
session = 1
directory = root / f"derivatives-session-{session}"
participants = [9, 12, 23, 28, 31, 32, 34, 36, 57, 58]

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
        dfs = dict(r_squared=dict(), error=dict(), offset=dict(), exponent=dict())
        for key_ in dfs:
            for key in keys:
                dfs[key_][key] = list()

    for tmin in np.arange(0, raw.times[-1], 60):
        results = parameterize_spectrum(raw, tmin, tmin + 60)
        for key_ in dfs:
            dfs[key_]["participant"].append(participant)
            dfs[key_]["times"].append(tmin)
            dfs[key_]["n_epochs"].append(0 if results is None else results[3])
        if results is not None:
            r_squared, error, aperiodic_params, _ = results
            # fill dataframes
            counter = 0
            for ch in eeg_ch_names:
                if ch in raw.info["bads"]:
                    for key_ in dfs:
                        dfs[key_][ch].append(np.nan)
                else:
                    dfs["r_squared"][ch].append(r_squared[counter])
                    dfs["error"][ch].append(error[counter])
                    dfs["offset"][ch].append(aperiodic_params[counter, 0])
                    dfs["exponent"][ch].append(aperiodic_params[counter, 1])
                    counter += 1
        else:
            for ch in eeg_ch_names:
                for key_ in dfs:
                    dfs[key_][ch].append(np.nan)

    del raw

df_r_squared = pd.DataFrame.from_dict(dfs["r_squared"])
df_error = pd.DataFrame.from_dict(dfs["error"])
df_offset = pd.DataFrame.from_dict(dfs["offset"])
df_exponent = pd.DataFrame.from_dict(dfs["exponent"])
