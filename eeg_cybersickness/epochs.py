# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mne import Epochs, find_events

from .triggers import load_triggers
from .utils._checks import check_type

if TYPE_CHECKING:
    from mne import BaseEpochs
    from mne.io import BaseRaw


def create_epochs(raw: BaseRaw, duration: float, overlap: float) -> BaseEpochs:
    """Create epochs based on the synthetic STI channel.

    Parameters
    ----------
    raw : Raw
        Preprocessed raw recording with a syntehtic STI channel.
    duration : float
        Duration of each epoch in seconds.
    overlap : float
        Duration of the overlap between epochs in seconds.
        Must be 0 <= overlap < duration.

    Returns
    -------
    epochs : Epochs
        All the created epochs.
    """
    check_type(duration, ("numeric",), "duration")
    check_type(overlap, ("numeric",), "overlap")
    if duration <= 0:
        raise ValueError("Argument 'duration' should be a strictly positive number.")
    if raw.info["sfreq"] * duration != np.round(raw.info["sfreq"] * duration):
        raise ValueError(
            "Argument 'duration' does not define a precise number of samples. "
            f"{duration} seconds corresponds to {raw.info['sfreq'] * duration} samples."
        )

    if overlap < 0:
        raise ValueError("Argument 'overlap' should be a strictly positive number.")
    if not np.isclose(
        (duration - overlap) * raw.info["sfreq"],
        np.round((duration - overlap) * raw.info["sfreq"]),
    ):
        raise ValueError(
            "Argument 'overlap' does not define a precise number of samples. "
            f"A duration of {duration} seconds with an overlap of {overlap} seconds "
            f"corresponds to {(duration - overlap) * raw.info['sfreq']} samples."
        )

    events = find_events(raw, stim_channel="STI")
    durations = np.diff(events[:, 0])
    events_ = np.empty(shape=(0, 3), dtype=np.int64)
    for event, event_duration in zip(events, durations):
        start = event[0]
        stop = start + event_duration
        stop -= int(raw.info["sfreq"] * duration)
        ts = np.arange(
            start, stop + 1, (duration - overlap) * raw.info["sfreq"]
        ).astype(int)
        events_ = np.vstack(
            (
                events_,
                np.c_[
                    ts,
                    np.zeros(ts.size, dtype=int),
                    event[2] * np.ones(ts.size, dtype=int),
                ],
            )
        )

    event_id = {
        key: value
        for key, value in load_triggers().items()
        if value in np.unique(events_[:, 2])
    }
    return Epochs(
        raw,
        events_,
        event_id,
        tmin=0,
        tmax=duration,
        baseline=None,
        picks="all",
        preload=True,
        reject_by_annotation=True,
    )
