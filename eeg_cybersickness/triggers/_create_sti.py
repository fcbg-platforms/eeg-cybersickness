# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import pandas as pd
from mne import create_info
from mne.io import RawArray

from ..utils._checks import check_type, ensure_path
from ..utils._docs import fill_doc
from ..utils.logs import logger
from . import load_triggers

if TYPE_CHECKING:
    from mne.io import BaseRaw


@fill_doc
def create_sti(
    raw: BaseRaw, session: int, rotation_axes: Tuple[str, ...]
) -> RawArray:
    """Create a synthetic trigger channel.

    Parameters
    ----------
    raw : Raw
        Raw recording with a ``"Stimulus/s1"`` annotation.
    %(session)s
    %(rotation_axes)s

    Returns
    -------
    stim : Raw
        MNE Raw object with a single "stim" channel containing the triggers
        for the rotation sequence played during that EEG recording.
    """
    event = find_event_onset(raw, in_samples=True)
    info = create_info(["STI"], sfreq=raw.info["sfreq"], ch_types="stim")
    data = np.zeros(shape=(1, raw.times.size))
    triggers = load_triggers()

    if session == 2:  # baseline
        data[0, event] = triggers["start"]
        return RawArray(data, info)

    rotation_axes = sorted(rotation_axes)
    # TODO: Replace with importlib-resources
    sequence_fname = (
        Path(__file__).parent
        / "sequences"
        / f"session{session}-{'-'.join(rotation_axes)}.csv"
    )
    sequence_trigger, sequence_duration = _load_sequence(sequence_fname)

    idx = event  # idx at which we start placing the triggers
    for trigger, duration in zip(sequence_trigger, sequence_duration):
        data[0, idx] = trigger
        idx += int(duration * raw.info["sfreq"])
        if data.size <= idx:
            logger.warning(
                "The entire rotation sequence could not be fitted in this "
                "recording."
            )
            break
    return RawArray(data, info)


def _load_sequence(fname: Union[str, Path]) -> Tuple[List[int], List[float]]:
    """Load sequence from a CSV file.

    Parameters
    ----------
    fname : path-like
        Path to the CSV file to load.

    Returns
    -------
    sequence_trigger : list of int
        List of consecutive triggers.
    sequence_duration : list of float
        List of duration between consecutive triggers.

    Notes
    -----
    If the duration is set to 0, the entry is skipped.
    If the angle amount is set to 0 or if the coordinates on all 3 axis is set
    to 0, the rotation is null and thus the trigger "none" is used.
    """
    fname = ensure_path(fname, must_exist=True)
    df = pd.read_csv(fname, index_col="Index")
    triggers = load_triggers()
    axes = ("Pitch", "Yaw", "Roll")

    sequence_trigger = []
    sequence_duration = []
    for _, row in df.iterrows():
        if row["duration"] == 0:
            continue

        if row["AngleAmout"] == 0 or all(row[key] == 0 for key in axes):
            sequence_trigger.append(triggers["none"])
            sequence_duration.append(row["duration"])
        else:
            rotation_axes = sorted(
                [key.lower() for key in axes if row[key] != 0]
            )
            sequence_trigger.append(triggers["_".join(rotation_axes)])
            sequence_duration.append(row["duration"])

    return sequence_trigger, sequence_duration


def find_event_onset(raw: BaseRaw, in_samples) -> Union[int, float]:
    """Find the paradigm event onset in the EEG recording.

    Parameters
    ----------
    raw : Raw
        Raw recording with a ``"Stimulus/s1"`` annotation.
    in_samples : bool
        If True, returns the onset in samples corrected for ``raw.first_samp``.
        If False, returns the onset in seconds.

    Returns
    -------
    event : int | float
        Event time in samples.
    """
    check_type(raw, (BaseRaw,), "raw")
    check_type(in_samples, (bool,), "in_samples")
    event = None
    for annotation in raw.annotations:
        if annotation["description"] == "Stimulus/s1":
            event = annotation["onset"]
            break
    if event is None:
        raise RuntimeError(
            "The onset stimuli was not found in the EEG recording."
        )
    if in_samples:
        event = int(event * raw.info["sfreq"] - raw.first_samp)
    return event
