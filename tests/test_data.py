"""Test for data generation."""

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.pv_forecast_analysis.data import _solar_generation, LOCAL_TZ, load_forecast


EXAMPLES_PATH = Path(__file__).parent / "examples"
_DATE_RANGE = pd.date_range(
    "2024-10-27 23:00",
    "2024-10-28 23:00",
    freq="15min",
    tz=LOCAL_TZ,
    inclusive="left",
    name="timestamp",
)
_REFERENCE = pd.DataFrame(
    {"Plant_01": [1.0] * len(_DATE_RANGE)},
    index=_DATE_RANGE,
)


@pytest.fixture(scope="module")
def mock_data() -> Path:
    with patch("src.pv_forecast_analysis.data.DATA_PATH", EXAMPLES_PATH) as mock_path:
        yield mock_path


@pytest.fixture
def fix_numpy_seed():
    np.random.seed(42)


def test_solar_generation_simple(fix_numpy_seed):
    """Test solar generation with one set of parameters."""
    assert 1.88 == round(_solar_generation(12, 30, 2), 2)


@pytest.mark.parametrize(
    "hour, minute, capacity, result",
    [
        (12, 30, 2, 1.88),
        (6, 45, 3, 0.56),
        (18, 15, 4, 0),
    ],
)
def test_solar_generation_with_parameter_set(
    hour, minute, capacity, result, fix_numpy_seed
):
    """Test solar generation with multiple parameters."""
    assert result == round(_solar_generation(hour, minute, capacity), 2)


def test_load_forecast(mock_data):
    """Test loading timeseries file with example file."""
    result = load_forecast(dt.date(2024, 10, 27))
    # the pd.read_csv function does not recognize the frequency automatically
    assert_frame_equal(result, _REFERENCE, check_freq=False)
