"""Dataset interactions."""
from pathlib import Path

import datetime as dt

import pytz

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

FORECAST_FILE_DATE_FORMAT = '%Y%m%d'

DATA_PATH = Path(__file__).parent.parent.parent / 'data'

MASTER_DATA_FILE = 'solar_plants_master_data.csv'
MEASUREMENT_FILE = 'solar_actual_measurements.csv'
FORECAST_FILE_TEMPLATE = 'solar_forecasts_{date}.csv'

COL_TSO = 'TSO'
COL_CAPACITY = 'Capacity_MW'
COL_PLANT = 'Plant'
COL_TIMESTAMP = 'timestamp'

TIMESTAMP_FORMAT = '%d.%m.%Y %H:%M'
LOCAL_TZ = pytz.timezone('Europe/Berlin')


def _load_timeseries_file(timeseries_file_path: Path) -> pd.DataFrame:
    """
    Load timeseries file and transform accordingly.

    :param timeseries_file_path: path
    :return:
    """
    timeseries = pd.read_csv(timeseries_file_path)
    timeseries[COL_TIMESTAMP] = (
        pd.to_datetime(timeseries[COL_TIMESTAMP], format=TIMESTAMP_FORMAT)
        .dt.tz_localize(LOCAL_TZ, ambiguous='infer')
    )
    timeseries = timeseries.set_index(COL_TIMESTAMP)
    return timeseries

def load_forecast(date: dt.date) -> pd.DataFrame:
    """
    Load PV forecasts.

    :param date: of forecast
    :return: forecast from `date`
    """
    forecast_file_path = DATA_PATH / FORECAST_FILE_TEMPLATE.format(date=date.strftime(FORECAST_FILE_DATE_FORMAT))
    return _load_timeseries_file(forecast_file_path)

def load_best_forecasts() -> pd.DataFrame:
    """Get best forecast for timestamp."""
    forecast_files = DATA_PATH.glob(FORECAST_FILE_TEMPLATE.format(date='*'))

    best_forecast = pd.DataFrame()
    for forecast_file in forecast_files:
        next_forecast = _load_timeseries_file(forecast_file)
        best_forecast = pd.concat([best_forecast, next_forecast[~next_forecast.index.isin(best_forecast.index)]])
        best_forecast.update(next_forecast)

    return best_forecast

def load_measurement() -> pd.DataFrame:
    """Load PV measurements."""
    return _load_timeseries_file(DATA_PATH / MEASUREMENT_FILE)


def load_master_data() -> pd.DataFrame:
    """Load plant master data."""
    return pd.read_csv(DATA_PATH / MASTER_DATA_FILE)


def _generate_error(length: int):
    """Generate error worsening to the end."""
    # Parameters
    num_timesteps = length
    base_error = 0.02  # 2% base error
    max_error = 0.20  # 20% max error on last day

    # Generate smooth error function
    time = np.arange(num_timesteps)  # Time index
    growth_factor = np.linspace(base_error, max_error, num_timesteps)  # Linear growth
    smooth_error = growth_factor * np.sin(time / random.uniform(25, 75))

    return smooth_error


def _solar_generation(hour, minute, capacity):
    """Simulates solar power output based on the hour of the day."""
    qh = hour * 4 + minute // 15
    peak_factor = np.sin((qh - 24) / 48 * np.pi)  # Peak around noon (hour 12)
    return max(0, capacity * peak_factor * np.random.uniform(0.8, 1.2))  # Add variability


def _generate_data() -> None:
    """Generate master data, measurements and forecast files."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(42)
    # Parameters
    num_plants = 40
    capacity_range = (1, 100)  # MW
    start_time = LOCAL_TZ.localize(datetime(2024, 10, 18))
    end_time = start_time + timedelta(days=8)
    freq_minutes = 15
    forecast_days = 5
    tsos = ['TSO_A', 'TSO_B', 'TSO_C', 'TSO_D']

    # Generate master data file
    plants = [f'Plant_{i:02d}' for i in range(1, num_plants + 1)]
    capacities = [round(random.uniform(*capacity_range), 1) for _ in range(num_plants)]
    plant_tso = [random.choice(tsos) for _ in range(num_plants)]
    master_data = pd.DataFrame({COL_PLANT: plants, COL_CAPACITY: capacities, COL_TSO: plant_tso})
    master_data.to_csv(DATA_PATH / MASTER_DATA_FILE, index=False)

    # Generate actual measurements with DST handling
    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{freq_minutes}min',
        inclusive='left',
    )

    actual_data = []
    for plant, capacity in zip(plants, capacities):
        production = [_solar_generation(ts.hour, ts.minute, capacity) for ts in timestamps]
        actual_data.extend(zip([plant] * len(timestamps), timestamps, production))

    actual_df = pd.DataFrame(actual_data, columns=['Plant', 'Timestamp', 'Actual_MW'])
    actual_df = actual_df.pivot(index='Timestamp', columns='Plant', values='Actual_MW')
    actual_df.index.name = COL_TIMESTAMP
    actual_df.index += pd.Timedelta(days=8)  # move to dst period
    actual_df = (
        pd.concat([actual_df, actual_df.iloc[-4:]])
        .set_index(
            pd.date_range(
                start=min(actual_df.index),
                freq=f"{freq_minutes}min",
                periods=len(actual_df)+4)
        )
    )
    export_actual_df = actual_df.copy(deep=True)
    export_actual_df = export_actual_df.set_index(export_actual_df.index.strftime(TIMESTAMP_FORMAT))
    export_actual_df.index.name = COL_TIMESTAMP
    export_actual_df.round(3).to_csv(DATA_PATH / MEASUREMENT_FILE)

    # Generate forecasts with worsening accuracy
    for day in range((end_time - start_time).days):
        start = min(actual_df.index) + timedelta(days=day)
        end = start + timedelta(days=forecast_days)

        forecast_data = actual_df[start:end].copy(deep=True)
        if start + timedelta(days=forecast_days-1) not in forecast_data.index:
            break
        for plant in forecast_data.columns:
            error = _generate_error(len(forecast_data))
            forecast_data[plant] *= (1-error)

        export_forecast_data = forecast_data.copy(deep=True)
        export_forecast_data = export_forecast_data.set_index(export_forecast_data.index.strftime(TIMESTAMP_FORMAT))
        export_forecast_data.index.name = COL_TIMESTAMP
        file_name = FORECAST_FILE_TEMPLATE.format(date=start.strftime(FORECAST_FILE_DATE_FORMAT))
        export_forecast_data.round(3).to_csv(DATA_PATH / file_name)


if __name__ == '__main__':
    # _generate_data()
    df = load_master_data()
    df = load_measurement()
    df = load_forecast(dt.date(2024, 10, 26))
    df = load_best_forecasts()
    print(df)