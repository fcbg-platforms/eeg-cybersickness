# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

from .triggers import load_triggers

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
    event_id = load_triggers()
