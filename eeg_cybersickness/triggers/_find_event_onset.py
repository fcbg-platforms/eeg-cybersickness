from typing import Union

from mne.io import BaseRaw

from ..utils._checks import check_type


def _find_event_onset(raw: BaseRaw, in_samples) -> Union[int, float]:
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
