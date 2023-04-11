# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from mne import create_info
from mne.io import RawArray, read_raw_brainvision

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw


def read_raw(vhdr_fname: Union[str, Path]) -> BaseRaw:
    """Load a raw recording.

    Parameters
    ----------
    vhdr_fname : path-like
        Path to the header file of the ANT recording in BrainVision format.

    Returns
    -------
    raw : Raw
        MNE raw instance, with the mastoids and the EOG channel dropped.

    Notes
    -----
    A synthetic trigger channel STI is added.
    """
    raw = read_raw_brainvision(vhdr_fname, preload=True)
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
