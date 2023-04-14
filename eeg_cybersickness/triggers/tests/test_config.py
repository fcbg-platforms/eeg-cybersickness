"""Test config.py"""

from pathlib import Path

import pytest

from eeg_cybersickness.triggers import load_triggers

fname_valid = Path(__file__).parent / "data" / "test_triggers.ini"
fname_invalid = Path(__file__).parent / "data" / "test_triggers_invalid.ini"


def test_load_triggers():
    """Test loading of triggers."""
    tdef = load_triggers(fname_valid)
    assert tdef["start"] == 1010101
    assert tdef["pitch"] == 101
    assert tdef["yaw"] == 201
    assert tdef["roll"] == 301
    assert tdef["none"] == 401
    assert tdef["pitch_roll"] == 102
    assert tdef["pitch_yaw"] == 202
    assert tdef["roll_yaw"] == 302
    assert tdef["pitch_roll_yaw"] == 10101

    with pytest.raises(ValueError, match="Key 'start' is missing"):
        load_triggers(fname_invalid)
