# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ._checks import check_type, ensure_path
from ._docs import fill_doc
from .logs import logger

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def get_raw_fname(
    root: Union[str, Path], participant: int, session: int
) -> str:
    """Get the file names from the participant and session.

    Parameters
    ----------
    %(root)s
    %(participant)s
    %(session)s

    Returns
    -------
    fname_eeg : Path
        Path to the EEG header file of the experiment recording.
    fname_biopac : Path
        Path to the ACQ biopac file of the experiment recording.
    """
    root = ensure_path(root, must_exist=True)
    check_type(participant, ("int",), "participant")
    check_type(session, ("int",), "session")
    if participant <= 0 or 100 <= participant:
        raise ValueError(
            "The participant ID should be set between 1 and 100. "
            f"{participant} is not valid."
        )
    if session <= 0 and 5 <= session:
        raise ValueError(
            "The session ID should be set between 1 and 4. "
            f"{session} is not valid."
        )

    fname_eeg = (
        root
        / "raw"
        / f"P{str(participant).zfill(2)}"
        / f"S{session}"
        / "eeg"
        / "experiment.vhdr"
    )
    if not fname_eeg.exists():
        logger.warning("The EEG file %s does not exist.", fname_eeg)
    fname_biopac = (
        root
        / "raw_aux"
        / f"P{str(participant).zfill(2)}"
        / f"S{session}"
        / "experiment.acq"
    )
    if not fname_biopac.exists():
        logger.warning("The Biopac file %s does not exist.", fname_biopac)
    return fname_eeg, fname_biopac
