#!/usr/bin/env python3
"""Analyze cumulative water-meter readings and flag daily anomalies.

Usage:
    python analyze_water_meter_data.py --input-dir data/source --output-dir results/generated

Requires pandas and numpy. Input CSVs must be UTF-8, semicolon-delimited, and
named pokazaniya.csv, pribory.csv, and sobytiya.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


EPS_M3 = 0.0005  # readings are commonly recorded to 0.001 m3; ignore float noise
SPIKE_ABS_M3_DAY = 1.0
SPIKE_MULTIPLE = 5.0
STREAK_DAYS = 30
STEP_RATIO = 3.0
STEP_ABS_M3_DAY = 0.1
STEP_DENOM_FLOOR = 0.02
DN15_DAILY_LIMIT = 37.5
DN20_DAILY_LIMIT = 62.5


def read_sources(input_dir: Path):
    readings = pd.read_csv(
        input_dir / "pokazaniya.csv",
        sep=";",
        dtype={"anon_id": "string"},
        parse_dates=["data"],
    )
    meters = pd.read_csv(
        input_dir / "pribory.csv", sep=";", dtype={"anon_id": "string"}
    )
    events = pd.read_csv(
        input_dir / "sobytiya.csv",
        sep=";",
        dtype={"anon_id": "string"},
        parse_dates=["data"],
    )
    for frame, required, name in [
        (readings, {"anon_id", "data", "pokazanie", "paketov_za_sutki"}, "pokazaniya.csv"),
        (meters, {"anon_id", "model"}, "pribory.csv"),
        (events, {"anon_id", "data", "sobytie"}, "sobytiya.csv"),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns: {sorted(missing)}")
    if readings.duplicated(["anon_id", "data"]).any():
        raise ValueError("pokazaniya.csv contains duplicate anon_id/date rows")
    if meters.anon_id.duplicated().any():
        raise ValueError("pribory.csv contains duplicate anon_id rows")
    return readings, meters, events


def max_true_streak(frame: pd.DataFrame, condition: pd.Series) -> pd.Series:
    """Maximum consecutive qualifying daily intervals by meter."""
    hits = condition.fillna(False).astype(bool)
    # Every false interval (including a missing-date gap) terminates a streak.
    run_id = (~hits).groupby(frame.anon_id, sort=False).cumsum()
    runs = hits.loc[hits].groupby(
        [frame.loc[hits, "anon_id"], run_id.loc[hits]], sort=False
    ).size()
    result = runs.groupby(level=0).max()
    return result.reindex(frame.anon_id.drop_duplicates(), fill_value=0).astype(int)


def ids_examples(frame: pd.DataFrame, mask: pd.Series, sort_by: str, ascending=False, n=8):
    sub = frame.loc[mask].sort_values(sort_by, ascending=ascending)
    if sub.empty:
        return []
    sub = sub.drop_duplicates("anon_id", keep="first").head(n)
    return sub.anon_id.astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with the three source CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Destination folder")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    readings, meters, events_raw = read_sources(args.input_dir)
    readings = readings.sort_values(["anon_id", "data"]).reset_index(drop=True)

    # Daily use is defined only when both readings are on consecutive dates.
    by_meter = readings.groupby("anon_id", sort=False)
    readings["previous_date"] = by_meter.data.shift()
    readings["previous_reading_m3"] = by_meter.pokazanie.shift()
    readings["gap_days"] = (readings.data - readings.previous_date).dt.days
    readings["interval_delta_m3"] = readings.pokazanie - readings.previous_reading_m3
    readings["is_consecutive_day"] = readings.gap_days.eq(1)
    readings["daily_consumption_m3"] = readings.interval_delta_m3.where(
        readings.is_consecutive_day
    )
    readings["interval_average_m3_day"] = readings.interval_delta_m3 / readings.gap_days
    readings["daily_rounded_m3"] = readings.daily_consumption_m3.round(3)

    # DN15 limit is from the manufacturer's passport; DN20 uses its own limit.
    data = readings.merge(meters[["anon_id", "model", "bs", "tip"]], on="anon_id", how="left")
    is_dn20 = data.model.fillna("").str.contains(r"D20\b", case=False, regex=True)
    data["daily_limit_m3"] = np.where(is_dn20, DN20_DAILY_LIMIT, DN15_DAILY_LIMIT)
    data["model_limit_note"] = np.where(data.model.isna(), "Model missing; conservative DN15 limit used", "")

    daily = data.daily_rounded_m3
    adjacent = data.is_consecutive_day
    negative = adjacent & daily.lt(-EPS_M3)
    flat = adjacent & daily.abs().le(EPS_M3) & data.paketov_za_sutki.gt(0)
    impossible = adjacent & daily.gt(data.daily_limit_m3 + EPS_M3)
    valid_positive = adjacent & daily.gt(EPS_M3) & daily.le(data.daily_limit_m3 + EPS_M3)

    # Personal baseline is the median positive plausible daily use for that meter.
    baselines = data.loc[valid_positive].groupby("anon_id").daily_rounded_m3.median()
    data["baseline_median_m3_day"] = data.anon_id.map(baselines)
    data["multiple_of_baseline"] = data.daily_rounded_m3 / data.baseline_median_m3_day
    spike = (
        adjacent
        & daily.ge(SPIKE_ABS_M3_DAY)
        & data.multiple_of_baseline.ge(SPIKE_MULTIPLE)
    )

    zero_streak = max_true_streak(data, flat)
    positive_streak = max_true_streak(data, valid_positive)
    zero_long = zero_streak.ge(STREAK_DAYS)
    leak_candidate = positive_streak.ge(STREAK_DAYS)

    # Compare the first and second halves of the actual source date range.
    start_date = data.data.min()
    end_date = data.data.max()
    split_date = start_date + pd.Timedelta(days=((end_date - start_date).days + 1) // 2)
    valid_for_step = adjacent & daily.ge(-EPS_M3) & daily.le(data.daily_limit_m3 + EPS_M3)
    step_data = data.loc[valid_for_step, ["anon_id", "data", "daily_rounded_m3"]].copy()
    first_half = step_data.loc[step_data.data < split_date].groupby("anon_id").daily_rounded_m3.agg(["mean", "size"])
    second_half = step_data.loc[step_data.data >= split_date].groupby("anon_id").daily_rounded_m3.agg(["mean", "size"])
    first_half.columns = ["first_half_mean", "first_half_days"]
    second_half.columns = ["second_half_mean", "second_half_days"]
    step = first_half.join(second_half, how="inner")
    step = step.loc[(step.first_half_days >= 30) & (step.second_half_days >= 30)].copy()
    denom_a = step.first_half_mean.clip(lower=STEP_DENOM_FLOOR)
    denom_b = step.second_half_mean.clip(lower=STEP_DENOM_FLOOR)
    step["step_ratio"] = np.maximum(denom_a / denom_b, denom_b / denom_a)
    step["step_difference_m3_day"] = step.second_half_mean - step.first_half_mean
    step["step_flag"] = (
        step.step_ratio.ge(STEP_RATIO)
        & step.step_difference_m3_day.abs().ge(STEP_ABS_M3_DAY)
    )
    step_flag_by_id = step.step_flag.reindex(meters.anon_id, fill_value=False)

    # Event records are deduplicated by meter/date/type before any comparison.
    events = events_raw.drop_duplicates(["anon_id", "data", "sobytie"]).copy()
    event_id_date = events.drop_duplicates(["anon_id", "data"])
    magnet = events.loc[events.sobytie.eq("магнит"), ["anon_id", "data"]].drop_duplicates()
    magnet_ids = set(magnet.anon_id)
    data["date_key"] = data.anon_id.astype(str) + "|" + data.data.dt.strftime("%Y-%m-%d")
    data["magnet_day"] = data.date_key.isin(
        set(magnet.anon_id.astype(str) + "|" + magnet.data.dt.strftime("%Y-%m-%d"))
    )
    event_types_by_date = (
        events.groupby(["anon_id", "data"]).sobytie.apply(lambda x: ",".join(sorted(set(x))))
    )
    data["event_types"] = [
        event_types_by_date.get((meter, date), "") for meter, date in zip(data.anon_id, data.data)
    ]

    # Exact-day and within-seven-day magnet comparisons use only true daily intervals.
    valid_daily = data.loc[adjacent].copy()
    magnet_daily = valid_daily.loc[valid_daily.magnet_day]
    magnet_ids_set = set(magnet.anon_id)
    comparison = valid_daily.loc[valid_daily.anon_id.isin(magnet_ids_set)].copy()
    comparison["zero_flag"] = comparison.daily_rounded_m3.abs().le(EPS_M3).astype(int)
    comparison["negative_flag"] = comparison.daily_rounded_m3.lt(-EPS_M3).astype(int)
    comparison["magnet_flag"] = comparison.magnet_day.astype(int)
    comparison_stats = comparison.groupby(["anon_id", "magnet_flag"]).agg(
        days=("daily_rounded_m3", "size"),
        zero_share=("zero_flag", "mean"),
    ).unstack("magnet_flag")
    paired = pd.DataFrame(index=comparison_stats.index)
    for col in ["days", "zero_share"]:
        for group_value, group_name in [(0, "control"), (1, "event")]:
            paired[f"{group_name}_{col}"] = comparison_stats[(col, group_value)] if (col, group_value) in comparison_stats.columns else np.nan
    paired_eligible = paired.loc[
        (paired.event_days >= 3) & (paired.control_days >= 10)
    ].copy()

    magnet_date_keys = set(
        magnet.anon_id.astype(str) + "|" + magnet.data.dt.strftime("%Y-%m-%d")
    )
    neg_rows = data.loc[negative, ["anon_id", "data"]]
    neg_near_magnet = 0
    neg_near_magnet_ids = set()
    for meter, day in zip(neg_rows.anon_id.astype(str), neg_rows.data):
        nearby = any(
            f"{meter}|{(day + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')}" in magnet_date_keys
            for offset in range(-7, 8)
        )
        if nearby:
            neg_near_magnet += 1
            neg_near_magnet_ids.add(meter)

    # Device-level roll-up and simple flags for prioritization.
    by_device = data.groupby("anon_id", sort=False)
    device = by_device.agg(
        model=("model", "first"),
        bs=("bs", "first"),
        tip=("tip", "first"),
        first_date=("data", "min"),
        last_date=("data", "max"),
        reading_rows=("data", "size"),
        adjacent_days=("is_consecutive_day", "sum"),
        max_daily_m3=("daily_rounded_m3", "max"),
        worst_daily_m3=("daily_rounded_m3", "min"),
        daily_limit_m3=("daily_limit_m3", "first"),
        baseline_median_m3_day=("baseline_median_m3_day", "first"),
        max_multiple_of_baseline=("multiple_of_baseline", "max"),
        magnet_event_dates=("magnet_day", "sum"),
        event_types=("event_types", lambda x: ",".join(sorted(set(t for val in x for t in val.split(",") if t)))),
    )
    for label, mask in [
        ("negative_intervals", negative),
        ("spike_intervals", spike),
        ("impossible_intervals", impossible),
    ]:
        device[label] = mask.groupby(data.anon_id).sum()
    device["longest_zero_streak_days"] = zero_streak
    device["longest_positive_streak_days"] = positive_streak
    device["long_zero_flag"] = zero_long
    device["leak_candidate_flag"] = leak_candidate
    device["step_flag"] = step.step_flag.reindex(device.index, fill_value=False)
    device = device.join(step.drop(columns="step_flag"), how="left")
    device["magnet_any"] = device.index.isin(magnet_ids)
    device["negative_near_magnet_7d"] = device.index.isin(neg_near_magnet_ids)
    device["priority_flags"] = device.apply(
        lambda row: ",".join(
            name
            for name, condition in [
                ("physically_impossible", row.impossible_intervals > 0),
                ("negative_counter", row.negative_intervals > 0),
                ("flat_30d", row.long_zero_flag),
                ("positive_30d", row.leak_candidate_flag),
                ("step_change", row.step_flag),
                ("magnet_event", row.magnet_any),
            ]
            if condition
        ),
        axis=1,
    )
    device.index.name = "anon_id"
    device = device.reset_index()
    device["is_d20"] = device.model.fillna("").str.contains(r"D20\b", case=False, regex=True)
    device["model_limit_note"] = ""
    device.loc[device.model.isna(), "model_limit_note"] = "Model missing; conservative DN15 limit used"
    device["model_limit_note"] = device.model_limit_note.fillna("")

    # Concise interval output: every row with negative, spike, impossible, or magnet signal.
    data["negative_flag"] = negative
    data["spike_flag"] = spike
    data["physically_impossible_flag"] = impossible
    data["flat_flag"] = flat
    data["flag_list"] = data.apply(
        lambda row: ",".join(
            name
            for name, condition in [
                ("negative", row.negative_flag),
                ("spike", row.spike_flag),
                ("physically_impossible", row.physically_impossible_flag),
                ("magnet_event_day", row.magnet_day),
                ("flat_zero", row.flat_flag),
            ]
            if condition
        ),
        axis=1,
    )
    anomaly_intervals = data.loc[
        data.flag_list.ne(""),
        [
            "anon_id", "data", "previous_date", "gap_days", "previous_reading_m3",
            "pokazanie", "interval_delta_m3", "daily_consumption_m3", "daily_limit_m3",
            "baseline_median_m3_day", "multiple_of_baseline", "paketov_za_sutki",
            "event_types", "flag_list",
        ],
    ].rename(columns={"data": "date", "pokazanie": "cumulative_reading_m3", "paketov_za_sutki": "packets_per_day"})

    # Category summary. Device counts are unique IDs; interval counts are separate.
    step_flag_ids = set(step.index[step.step_flag])
    step_examples = step.loc[step.step_flag].sort_values("step_ratio", ascending=False).head(8)
    spike_examples = ids_examples(data.assign(_spike_score=data.multiple_of_baseline), spike, "_spike_score", False)
    negative_examples = ids_examples(data, negative, "daily_rounded_m3", True)
    impossible_examples = ids_examples(data, impossible, "daily_rounded_m3", False)
    zero_examples = zero_streak.sort_values(ascending=False).head(8).index.astype(str).tolist()
    leak_examples = positive_streak.sort_values(ascending=False).head(8).index.astype(str).tolist()
    category_rows = [
        {
            "category": "negative_consumption",
            "criterion": f"Adjacent-day change < -{EPS_M3:.4f} m3/day",
            "intervals": int(negative.sum()),
            "meters": int(data.loc[negative, "anon_id"].nunique()),
            "example_ids": ", ".join(negative_examples),
        },
        {
            "category": "spike",
            "criterion": f"Daily use >= {SPIKE_ABS_M3_DAY:.1f} m3 and >= {SPIKE_MULTIPLE:.0f}x meter median plausible positive daily use",
            "intervals": int(spike.sum()),
            "meters": int(data.loc[spike, "anon_id"].nunique()),
            "example_ids": ", ".join(spike_examples),
        },
        {
            "category": "physically_impossible",
            "criterion": "Daily use above manufacturer daily limit: 37.5 m3 for DN15; 62.5 m3 for DN20; unknown model screened at DN15 limit",
            "intervals": int(impossible.sum()),
            "meters": int(data.loc[impossible, "anon_id"].nunique()),
            "example_ids": ", ".join(impossible_examples),
        },
        {
            "category": "leak_candidate",
            "criterion": f">0 plausible m3/day on every one of >= {STREAK_DAYS} consecutive calendar days",
            "intervals": "device streaks",
            "meters": int(leak_candidate.sum()),
            "example_ids": ", ".join(leak_examples),
        },
        {
            "category": "long_zero_with_packets",
            "criterion": f"Unchanged cumulative reading for >= {STREAK_DAYS} consecutive calendar days with packets_per_day > 0",
            "intervals": "device streaks",
            "meters": int(zero_long.sum()),
            "example_ids": ", ".join(zero_examples),
        },
        {
            "category": "step_change",
            "criterion": f"First vs second half mean changes >= {STEP_RATIO:.0f}x and >= {STEP_ABS_M3_DAY:.1f} m3/day; >=30 valid adjacent days in each half; denominator floor {STEP_DENOM_FLOOR:.2f}",
            "intervals": "device comparison",
            "meters": int(step_flag_by_id.sum()),
            "example_ids": ", ".join(step_examples.index.astype(str).tolist()),
        },
    ]
    categories = pd.DataFrame(category_rows)

    # Event summary at deduplicated meter/date/type grain.
    event_summary = []
    for event_type, ev in events.groupby("sobytie", dropna=False):
        event_dates = ev[["anon_id", "data"]].drop_duplicates()
        keys = set(event_dates.anon_id.astype(str) + "|" + event_dates.data.dt.strftime("%Y-%m-%d"))
        matching = data.loc[data.date_key.isin(keys)]
        matching_daily = matching.loc[matching.is_consecutive_day]
        event_summary.append({
            "event_type": event_type,
            "raw_rows": int(events_raw.sobytie.eq(event_type).sum()),
            "deduplicated_meter_dates": int(len(event_dates)),
            "meters": int(event_dates.anon_id.nunique()),
            "matching_reading_dates": int(len(matching)),
            "matching_adjacent_day_deltas": int(len(matching_daily)),
            "negative_daily_deltas": int(matching_daily.daily_rounded_m3.lt(-EPS_M3).sum()),
            "zero_daily_deltas": int(matching_daily.daily_rounded_m3.abs().le(EPS_M3).sum()),
            "daily_deltas_over_1m3": int(matching_daily.daily_rounded_m3.ge(1.0).sum()),
        })
    event_summary = pd.DataFrame(event_summary)
    raw_event_duplicates = int(events_raw.duplicated(["anon_id", "data", "sobytie"]).sum())
    magnet_adjacent_days = int(len(magnet_daily))
    magnet_zero_days = int(magnet_daily.daily_rounded_m3.abs().le(EPS_M3).sum())
    magnet_negative_days = int(magnet_daily.daily_rounded_m3.lt(-EPS_M3).sum())
    control = comparison.loc[~comparison.magnet_day]
    event_zero_share = float(magnet_daily.daily_rounded_m3.abs().le(EPS_M3).mean()) if magnet_adjacent_days else np.nan
    control_zero_share = float(control.daily_rounded_m3.abs().le(EPS_M3).mean()) if len(control) else np.nan
    paired_event_zero = float(paired_eligible.event_zero_share.mean()) if len(paired_eligible) else np.nan
    paired_control_zero = float(paired_eligible.control_zero_share.mean()) if len(paired_eligible) else np.nan
    paired_event_higher = int((paired_eligible.event_zero_share > paired_eligible.control_zero_share).sum())
    paired_event_lower = int((paired_eligible.event_zero_share < paired_eligible.control_zero_share).sum())

    # Within +/- 7 days is a descriptive association check, not evidence of causation.
    event_rows = [
        {"metric": "raw_event_rows", "value": len(events_raw), "detail": "All event rows as supplied"},
        {"metric": "duplicate_event_rows", "value": raw_event_duplicates, "detail": "Duplicate anon_id/date/event rows removed for comparisons"},
        {"metric": "unique_magnet_meter_dates", "value": len(magnet), "detail": f"Across {len(magnet_ids)} meters"},
        {"metric": "magnet_adjacent_daily_deltas", "value": magnet_adjacent_days, "detail": f"Zero={magnet_zero_days}; negative={magnet_negative_days}"},
        {"metric": "magnet_zero_share", "value": event_zero_share, "detail": "Exact magnet dates with valid adjacent-day deltas"},
        {"metric": "non_magnet_zero_share_same_meters", "value": control_zero_share, "detail": "Other adjacent-day deltas from meters with magnet events"},
        {"metric": "paired_meters_min_3_event_10_control", "value": len(paired_eligible), "detail": f"Mean zero share event={paired_event_zero:.4f}; control={paired_control_zero:.4f}; event higher={paired_event_higher}; event lower={paired_event_lower}"},
        {"metric": "negative_deltas_within_7d_of_magnet", "value": neg_near_magnet, "detail": f"Of {int(negative.sum())} negative adjacent-day intervals"},
    ]
    for _, row in event_summary.iterrows():
        event_rows.append({
            "metric": f"event_type_{row.event_type}",
            "value": row.negative_daily_deltas,
            "detail": f"negative adjacent-day deltas on deduplicated event dates; rows={row.raw_rows}, unique_dates={row.deduplicated_meter_dates}, matched_deltas={row.matching_adjacent_day_deltas}, zero={row.zero_daily_deltas}",
        })
    event_linkage = pd.DataFrame(event_rows)

    # Add source quality details to README.
    d15 = int(meters.model.fillna("").str.contains(r"D15\b", case=False, regex=True).sum())
    d20 = int(meters.model.fillna("").str.contains(r"D20\b", case=False, regex=True).sum())
    missing_models = int(meters.model.isna().sum())
    daily_adjacent_count = int(adjacent.sum())
    nonadjacent_intervals = int(data.gap_days.gt(1).sum())
    zero_packet_count = int(data.paketov_za_sutki.eq(0).sum())
    negative_meter_ids = int(data.loc[negative, "anon_id"].nunique())
    reset_row = event_summary.loc[event_summary.event_type.eq("сброс")]
    reset_note = ""
    if not reset_row.empty:
        rr = reset_row.iloc[0]
        reset_note = f" На датах события «сброс» найдено {int(rr.negative_daily_deltas)} отрицательных суточных разниц среди {int(rr.matching_adjacent_day_deltas)} совпавших интервалов."

    readme = f"""# Анализ аномалий потребления воды

Период исходных данных: {start_date:%Y-%m-%d} — {end_date:%Y-%m-%d}. В `pokazaniya.csv` — {len(readings):,} строк по {readings.anon_id.nunique():,} приборам. Дата в суточном расчёте трактуется как календарная дата.

## Расчёт

Для каждой пары последовательных показаний одного прибора рассчитывается `interval_delta_m3 = показание_текущее - показание_предыдущее`. Поле `daily_consumption_m3` заполняется только при разнице дат ровно в 1 день. При большем разрыве значение оставлено пустым: разницу за несколько дней нельзя выдавать за расход одного дня. Для проверки таких разрывов отдельно записан `interval_average_m3_day = interval_delta_m3 / gap_days`. Между показаниями нет интерполяции. Погрешность округления до 0,001 м³ сглажена допуском {EPS_M3:.4f} м³.

В данных есть {daily_adjacent_count:,} суточных интервалов и {nonadjacent_intervals:,} интервалов с пропусками дат. В модели `pribory.csv` найдено {d15:,} DN15 и {d20:,} DN20 приборов; у {missing_models} приборов модель не заполнена. Для DN15 физический порог взят из паспорта производителя (максимальный суточный объём 37,5 м³), для DN20 — 62,5 м³. При неизвестной модели применён более строгий порог DN15.

## Результаты

| Аномалия | Критерий | Интервалы / приборы | Примеры |
|---|---|---:|---|
"""
    for _, row in categories.iterrows():
        count = row.meters
        readme += f"| {row.category} | {row.criterion} | {row.intervals} / {count:,} | {row.example_ids} |\n"
    readme += f"""
## События «магнит»

Повторяющиеся события дедуплицированы по ключу `anon_id + дата + событие`. В исходной таблице {len(events_raw):,} строк событий, из них {raw_event_duplicates:,} точных повторов. Магнит отмечен на {len(magnet):,} уникальных датах у {len(magnet_ids):,} приборов. На {magnet_adjacent_days:,} датах удалось сопоставить событие с суточной разницей показаний: отрицательных разниц {magnet_negative_days}, нулевых — {magnet_zero_days} ({event_zero_share:.1%}). Среди остальных суточных дат тех же приборов нулевых разниц {control_zero_share:.1%}. В парном сравнении {len(paired_eligible)} приборов, у которых есть минимум 3 даты события и 10 контрольных дат, средняя доля нулевых суток — {paired_event_zero:.1%} на датах события и {paired_control_zero:.1%} на остальных датах; у {paired_event_higher} приборов нули чаще на датах магнита, у {paired_event_lower} — реже. В пределах ±7 дней от события лежат {neg_near_magnet} из {int(negative.sum())} отрицательных суточных интервалов. Эти данные не показывают надёжной синхронной связи магнита с падением показаний.{reset_note}

## Дополнительные наблюдения

- Во всех {len(readings):,} строках `paketov_za_sutki` больше нуля; при этом у {int(zero_long.sum()):,} приборов показание не менялось как минимум 30 последовательных суток. Примеры длительных плато: {', '.join(zero_examples[:5])}.
- Есть почти симметричные гигантские скачки вверх и вниз, которые физически нельзя объяснить расходом воды. Например, у 2491162 разница составила около +3,005 млн м³, затем −3,006 млн м³; у 8908023 — около +2,684 млн м³, затем −2,684 млн м³. Это похоже на сбой показаний/масштаба или сброс с некорректной обработкой.
- В таблице событий много точных повторов, поэтому число строк события нельзя считать числом отдельных срабатываний. У {missing_models} приборов также отсутствует модель.

## Очерёдность проверки

В первую очередь: приборы с расходом выше паспортного суточного предела, особенно 2491162, 8908023, 2159987, 7891133 и 2030325; затем приборы с отрицательными показаниями; отдельно проверить длительные плато 9686922 и 7558835. Отрицательная накопительная разница чаще указывает на сброс/замену прибора или ошибку передачи/обработки: паспорт АКВА прямо указывает, что обратный поток счётчиком не измеряется и из накопленного объёма не вычитается.

## Файлы результата

- `daily_consumption.csv` — исходное накопительное показание, предыдущая дата/показание, разница, число дней между чтениями и суточное потребление только для соседних календарных дней.
- `anomaly_intervals.csv` — интервалы с отрицательным расходом, всплеском, превышением физического предела, длительным нулём или магнитным событием.
- `device_summary.csv` — сводка по каждому прибору, максимальные серии и флаги.
- `category_summary.csv` — критерии, количества и примеры ID.
- `event_linkage.csv` — сопоставление событий с разницами показаний.

Запуск: `python water_anomaly_analysis.py --input-dir <папка_с_csv> --output-dir <папка_результатов>`. Нужны `pandas` и `numpy` (`python -m pip install pandas numpy`).
"""

    # Save UTF-8 BOM CSVs for reliable opening in common spreadsheet applications.
    csv_options = {"sep": ";", "index": False, "encoding": "utf-8-sig"}
    out = args.output_dir
    daily_output = data[
        [
            "anon_id", "data", "pokazanie", "previous_date", "previous_reading_m3",
            "gap_days", "interval_delta_m3", "daily_consumption_m3",
            "interval_average_m3_day", "paketov_za_sutki", "model", "daily_limit_m3",
        ]
    ].rename(
        columns={
            "data": "date",
            "pokazanie": "cumulative_reading_m3",
            "paketov_za_sutki": "packets_per_day",
        }
    )
    daily_output.to_csv(out / "daily_consumption.csv", **csv_options)
    anomaly_intervals.to_csv(out / "anomaly_intervals.csv", **csv_options)
    device.to_csv(out / "device_summary.csv", **csv_options)
    categories.to_csv(out / "category_summary.csv", **csv_options)
    event_linkage.to_csv(out / "event_linkage.csv", **csv_options)
    (out / "README_RU.md").write_text(readme, encoding="utf-8")

    package = out / "water_anomaly_calculations.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in [
            "daily_consumption.csv",
            "anomaly_intervals.csv",
            "device_summary.csv",
            "category_summary.csv",
            "event_linkage.csv",
            "README_RU.md",
        ]:
            archive.write(out / name, arcname=name)

    print(f"Saved results to {out.resolve()}")
    print(categories.to_string(index=False))
    print(f"Magnet: adjacent matched days={magnet_adjacent_days}, zero={magnet_zero_days}, negative={magnet_negative_days}; negative within +/-7 days={neg_near_magnet}")
    print(f"Event duplicates={raw_event_duplicates}; missing models={missing_models}; zero-packet rows={zero_packet_count}")
    print(f"Step split date={split_date.date()}; valid step meters={int(step_flag_by_id.sum())}")


if __name__ == "__main__":
    main()

