"""Dataset interactions."""
from pathlib import Path

import datetime as dt

import pytz

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


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
    plants = [f'Plant_{i}' for i in range(1, num_plants + 1)]
    capacities = [random.uniform(*capacity_range) for _ in range(num_plants)]
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
        export_forecast_data.round(3).to_csv(DATA_PATH / FORECAST_FILE_TEMPLATE.format(date=start.strftime('%Y%m%d')))


if __name__ == '__main__':
    _generate_data()