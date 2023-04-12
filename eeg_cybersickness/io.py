# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from bioread import read
from mne import create_info
from mne.io import RawArray, read_raw_brainvision

from .utils._checks import ensure_path

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw


def read_raw_eeg(fname_vhdr: Union[str, Path]) -> BaseRaw:
    """Load a raw recording.

    Parameters
    ----------
    fname_vhdr : path-like
        Path to the header file of the ANT recording in BrainVision format.

    Returns
    -------
    raw : Raw
        MNE raw instance, with the mastoids and the EOG channel dropped.

    Notes
    -----
    A synthetic trigger channel STI is added.
    """
    fname_vhdr = ensure_path(fname_vhdr, must_exist=True)
    raw = read_raw_brainvision(fname_vhdr, preload=True)
    raw.drop_channels(["M1", "M2", "EOG"])

    # create stim channel
    assert raw.annotations[2]["description"] == "Stimulus/s1"
    onset = int(raw.annotations[2]["onset"] * raw.info["sfreq"])
    info = create_info(["STI"], sfreq=raw.info["sfreq"], ch_types="stim")
    data = np.zeros(shape=(1, raw.times.size))

    # Let's figure out where we need to put ones in the array. We need to build
    # the sequence which corresponds to (5s x 12 + 4s x 1) x 20.
    sequence = np.empty(13)
    sequence[:-1] = np.arange(
        0, 5 * 12 * raw.info["sfreq"], 5 * raw.info["sfreq"]
    )
    sequence[-1] = 64 * raw.info["sfreq"]
    sequence = sequence.astype(int)
    # gives us one of the 20 sequences and needs to start at the onset and
    # needs to be repeated every 64 seconds.
    for k in range(20):
        for elt in sequence:
            idx = int(onset + k * 64 * raw.info["sfreq"] + elt)
            data[0, idx] = 1

    stim = RawArray(data, info)
    raw.add_channels([stim], force_update_info=True)
    return raw


def read_raw_biopac(fname_biopac: Union[str, Path]) -> BaseRaw:
    """Load an ACQ biopac recording.

    Parameters
    ----------
    fname_biopac : path-like
        Path to the ACQ file of the Biopac recording.

    Returns
    -------
    raw : Raw
        MNE raw instance, with 3 channels: ECG, EGG and STI-Biopac.
    """

    def data_or_blank(channel, index, missing_val=0):
        """Safe data loader.

        Taken from bioread.
        """
        ci = index // channel.frequency_divider
        if index % channel.frequency_divider == 0 and ci < channel.point_count:
            return channel.data[ci]
        return missing_val

    fname_biopac = str(ensure_path(fname_biopac, must_exist=True))
    data = read(fname_biopac)
    fs = data.samples_per_second  # Hz
    if len(data.channels) != 3:
        raise RuntimeError(
            "The reader expected 3 channels on the Biopac recording. "
            f"{len(data.channels)} found."
        )

    # retrieve data as an (n_channels, n_samples) array
    data_array = np.empty((len(data.channels), data.time_index.size))
    for i, t in enumerate(data.time_index):
        data_array[:, i] = [data_or_blank(c, i) for c in data.channels]

    # retrieve channel names and convert to Volts if necesary
    ch_names = []
    ch_types = []
    for k, channel in enumerate(data.channels):
        if "digital" in channel.name.lower():
            ch_names.append("STI-Biopac")
            ch_types.append("stim")
        elif "ecg" in channel.name.lower():
            ch_names.append("ECG")
            ch_types.append("ecg")
        elif "egg" in channel.name.lower():
            ch_names.append("EGG")
            ch_types.append("misc")

        if channel.units.lower().strip() == "mv":
            data_array[k, :] *= 1e-3  # convert to Volts

    # create MNE RawArray
    info = create_info(ch_names, fs, ch_types)
    raw = RawArray(data_array, info)
    return raw
