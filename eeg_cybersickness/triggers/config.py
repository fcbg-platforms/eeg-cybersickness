# postponed evaluation of annotations, c.f. PEP 563 and PEP 649
# alternatively, the type hints can be defined as strings which will be
# evaluated with eval() prior to type checking.
from __future__ import annotations

from configparser import ConfigParser
from importlib.resources import files
from typing import TYPE_CHECKING, Dict, Union

from ..utils._checks import ensure_path

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_TRIGGERS = files("eeg_cybersickness.triggers") / "triggers.ini"


def load_triggers(
    fname: Union[str, Path] = _DEFAULT_TRIGGERS,
) -> Dict[str, int]:
    """Load triggers from ``triggers.ini``.

    Parameters
    ----------
    fname : str | Path
        Path to the configuration file.
        Default to ``'eeg_cybersickness/triggers/triggers.ini'``.

    Returns
    -------
    triggers : dict
        Trigger definitiopn containing: start, none, pitch, roll, yaw,
        pitch_yaw, pitch_roll, roll_yaw, pitch_roll_yaw.
    """
    fname = ensure_path(fname, must_exist=True)
    config = ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.read(str(fname))

    triggers = dict()
    for name, value in config.items("events"):
        triggers[name] = int(value)

    # verification
    keys = (
        "start",
        "none",
        "pitch",
        "roll",
        "yaw",
        "pitch_roll",
        "pitch_yaw",
        "roll_yaw",
        "pitch_roll_yaw",
    )
    for key in keys:
        if key not in triggers:
            raise ValueError(f"Key '{key}' is missing from trigger definition file.")

    return triggers
