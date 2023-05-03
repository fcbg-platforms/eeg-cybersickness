# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from os import makedirs
from typing import TYPE_CHECKING

from ._checks import check_type, ensure_path
from ._docs import fill_doc
from .logs import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Tuple, Union


@fill_doc
def get_raw_fname(
    root: Union[str, Path], participant: int, session: int
) -> Tuple[Path, Path]:
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
    root, _, _ = _check_root_participant_session(root, participant, session)
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


@fill_doc
def get_derivative_stem(root: Union[str, Path], participant: int, session: int) -> Path:
    """Get the derivative file name stem from the participant and session.

    Parameters
    ----------
    %(root)s
    %(participant)s
    %(session)s

    Returns
    -------
    fname_stem : Path
        Path of the file name stem corresponding to the participant and
        session.

    Notes
    -----
    This function automatically created the participant folder if it does not
    exist.
    """
    root, _, _ = _check_root_participant_session(root, participant, session)
    fname_stem = (
        root
        / "derivatives"
        / f"P{str(participant).zfill(2)}"
        / f"P{str(participant).zfill(2)}_S{session}"
    )
    makedirs(fname_stem.parent, exist_ok=True)
    return fname_stem


@fill_doc
def _check_root_participant_session(
    root: Union[str, Path], participant: int, session: int
) -> Tuple[Path, int, int]:
    """Check the root, participant and session variables.

    Parameters
    ----------
    %(root)s
    %(participant)s
    %(session)s

    Returns
    -------
    root : Path
        Path to the folder containing ``"raw"`` (EEG recordings),
        ``"raw_aux"`` (Biopac recoridngs), and ``derivatives``.
    %(participant)s
    %(session)s
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
            "The session ID should be set between 1 and 4. " f"{session} is not valid."
        )
    return root, participant, session
