from configparser import ConfigParser
from pathlib import Path
from typing import Dict, Union

from ..utils._checks import ensure_path


def load_triggers(
    fname: Union[str, Path] = Path(__file__).parent / "triggers.ini"
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
            raise ValueError(
                f"Key '{key}' is missing from trigger definition file."
            )

    return triggers
