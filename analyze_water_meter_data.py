"""Calculate daily consumption from cumulative water-meter readings."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    readings = pd.read_csv(
        args.input_dir / "pokazaniya.csv",
        sep=";",
        dtype={"anon_id": "string"},
        parse_dates=["data"],
    )
    readings = readings.sort_values(["anon_id", "data"]).reset_index(drop=True)

    by_meter = readings.groupby("anon_id", sort=False)
    readings["previous_date"] = by_meter["data"].shift()
    readings["previous_reading_m3"] = by_meter["pokazanie"].shift()
    readings["gap_days"] = (readings["data"] - readings["previous_date"]).dt.days
    readings["interval_delta_m3"] = readings["pokazanie"] - readings["previous_reading_m3"]

    # Оставляем суточный расход пустым, если показания разделены пропуском дат.
    is_daily = readings["gap_days"].eq(1)
    readings["daily_consumption_m3"] = readings["interval_delta_m3"].where(is_daily)
    readings["interval_average_m3_day"] = readings["interval_delta_m3"] / readings["gap_days"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "anon_id",
        "data",
        "previous_date",
        "gap_days",
        "previous_reading_m3",
        "pokazanie",
        "interval_delta_m3",
        "daily_consumption_m3",
        "interval_average_m3_day",
    ]
    readings[columns].to_csv(
        args.output_dir / "daily_consumption.csv",
        sep=";",
        index=False,
        date_format="%Y-%m-%d",
    )


if __name__ == "__main__":
    main()
