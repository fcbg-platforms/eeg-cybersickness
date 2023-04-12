"""Test config.py"""

from pathlib import Path

import pytest

from eeg_cybersickness.config import load_triggers

fname_valid = Path(__file__).parent / "data" / "test_triggers.ini"
fname_invalid = Path(__file__).parent / "data" / "test_triggers_invalid.ini"


def test_load_triggers():
    """Test loading of triggers."""
    tdef = load_triggers(fname_valid)
    assert tdef["pitch"] == 101
    assert tdef["yaw"] == 201
    assert tdef["roll"] == 301
    assert tdef["pitch_yaw"] == 102
    assert tdef["pitch_roll"] == 202
    assert tdef["yaw_roll"] == 302
    assert tdef["pitch_yaw_roll"] == 10101

    with pytest.raises(ValueError, match="Key 'novel' is missing"):
        load_triggers(fname_invalid)
