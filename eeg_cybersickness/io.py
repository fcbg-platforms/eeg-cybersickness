# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from bioread import read
from mne import create_info, find_events
from mne.io import RawArray, read_raw_brainvision

from .triggers._create_sti import create_sti, find_event_onset
from .utils._checks import check_rotation_axes, ensure_path
from .utils._docs import fill_doc
from .utils.path import get_raw_fname

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw


@fill_doc
def read_raw(
    root: Union[str, Path],
    participant: int,
    session: int,
    rotation_axes: Tuple[str, ...] = ("Pitch", "Yaw", "Roll"),
) -> BaseRaw:
    """Load a raw recording.

    Parameters
    ----------
    %(root)s
    %(participant)s
    %(session)s
    %(rotation_axes)s

    Returns
    -------
    raw : Raw
        MNE raw recording with synchronize ECG/EGG and a synthetic trigger
        channel.
    """
    fname_eeg, fname_biopac = get_raw_fname(root, participant, session)
    check_rotation_axes(rotation_axes)
    raw_eeg = _read_raw_eeg(fname_eeg)
    raw_biopac = _read_raw_biopac(fname_biopac)

    # find onsets
    events_eeg = find_event_onset(raw_eeg, in_samples=False)
    events_biopac = find_events(raw_biopac)[0, 0] / raw_biopac.info["sfreq"]
    assert 0.2 <= events_biopac and 0.2 <= events_eeg
    raw_eeg.crop(events_eeg - 0.2, None)
    raw_biopac.crop(events_biopac - 0.2, None)
    raw_biopac.resample(raw_eeg.info["sfreq"])

    # figure out which one is longer and crop to the same size
    if raw_biopac.times[-1] < raw_eeg.times[-1]:
        raw_eeg.crop(0, raw_biopac.times[-1], include_tmax=True)
    else:
        raw_biopac.crop(0, raw_eeg.times[-1], include_tmax=True)

    # create synthetic trigger channel
    sti = create_sti(raw_eeg, session, rotation_axes)

    # concatenate
    raw_biopac.drop_channels(["STI-Biopac"])
    raw_eeg.add_channels([raw_biopac, sti], force_update_info=True)
    raw_eeg.set_annotations(None)

    return raw_eeg


def _read_raw_eeg(fname_vhdr: Union[str, Path]) -> BaseRaw:
    """Load a raw recording.

    Parameters
    ----------
    fname_vhdr : path-like
        Path to the header file of the ANT recording in BrainVision format.

    Returns
    -------
    raw : Raw
        MNE raw instance, with the mastoids and the EOG channel dropped.
    """
    fname_vhdr = ensure_path(fname_vhdr, must_exist=True)
    raw = read_raw_brainvision(fname_vhdr, preload=True)
    raw.drop_channels(["M1", "M2", "EOG"])
    return raw


def _read_raw_biopac(fname_biopac: Union[str, Path]) -> BaseRaw:
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

    # retrieve channel names and convert to Volts if necessary
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
