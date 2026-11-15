"""Forecast household electricity spend for the London smart meter dataset.

Method (deliberately simple, no ML):
  1. Seasonal profile: each household's mean daily kWh by calendar month x day type.
  2. Level factor: recent 8 weeks of actuals / what the profile expected for those
     weeks, so the forecast follows any recent change in usage.
  3. Forecast daily kWh = profile x level factor.
Spend = kWh x dummy unit rate (p/kWh) + dummy standing charge (p/day).
ToU households are on dummy Economy 7 / Economy 10 tariffs instead: their daily
kWh is split into peak and off-peak using their half-hourly history.

Outputs a Power BI-ready star schema of CSVs in ./output.
Run: python energy_forecast.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "archive"
HHBLOCK_DIR = DATA_DIR / "hhblock_dataset" / "hhblock_dataset"
OUTPUT_DIR = ROOT / "output"

CALENDAR_START = pd.Timestamp("2011-11-01")
CALENDAR_END = pd.Timestamp("2015-02-28")
BACKTEST_TRAIN_END = pd.Timestamp("2013-11-30")
BACKTEST_END = pd.Timestamp("2014-02-28")
FORECAST_TRAIN_END = pd.Timestamp("2014-02-28")
FORECAST_END = pd.Timestamp("2015-02-28")

MIN_READINGS_PER_DAY = 46       # drop days with more than 2 missing half-hours
ACTIVE_WINDOW_DAYS = 28         # household must have data this close to train end
LEVEL_WINDOW_DAYS = 56          # recent window used for the level factor
LEVEL_CLIP = (0.5, 2.0)
MIN_COHORT_HOUSEHOLDS = 30      # smaller Acorn groups use the overall seasonal index

RANDOM_SEED = 42
UNIT_RATE_RANGE_P_PER_KWH = (20.0, 35.0)
STANDING_CHARGE_RANGE_P_PER_DAY = (40.0, 65.0)

# Dummy two-rate tariffs for ToU households: (peak range, off-peak range) in p/kWh
# and off-peak windows in local clock time as [start, end).
TOU_TARIFFS = {
    "Economy 7": {
        "peak_range": (28.0, 38.0),
        "off_peak_range": (12.0, 18.0),
        "off_peak_windows": [("00:30", "07:30")],
    },
    "Economy 10": {
        "peak_range": (28.0, 36.0),
        "off_peak_range": (16.0, 22.0),
        "off_peak_windows": [("00:00", "05:00"), ("13:00", "16:00"), ("20:00", "22:00")],
    },
}

# England & Wales bank holidays. The supplied uk_bank_holidays.csv has several
# day/month swaps (e.g. 2012-07-05 for 2012-05-07) and stops at 2014.
BANK_HOLIDAYS = pd.to_datetime([
    "2011-12-26", "2011-12-27",
    "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04",
    "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26",
    "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27",
    "2013-08-26", "2013-12-25", "2013-12-26",
    "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-05", "2014-05-26",
    "2014-08-25", "2014-12-25", "2014-12-26",
    "2015-01-01",
])


def load_daily():
    """Daily kWh per household, keeping only (near-)complete days."""
    daily = pd.read_csv(
        DATA_DIR / "daily_dataset.csv",
        usecols=["LCLid", "day", "energy_sum", "energy_count"],
        dtype={"LCLid": "string"},
        parse_dates=["day"],
    )
    daily = daily.dropna(subset=["energy_sum"])
    daily = daily[daily["energy_count"] >= MIN_READINGS_PER_DAY]
    # Scale days with a missing reading or two up to a full 48 half-hours
    daily["kwh"] = daily["energy_sum"] * 48 / daily["energy_count"].clip(upper=48)
    daily = daily.rename(columns={"day": "date"})[["LCLid", "date", "kwh"]]
    return daily.drop_duplicates(["LCLid", "date"]).reset_index(drop=True)


def load_households():
    households = pd.read_csv(DATA_DIR / "informations_households.csv", dtype="string")
    households = households.rename(columns={"file": "block"})
    return households.sort_values("LCLid").reset_index(drop=True)


def build_calendar():
    dates = pd.date_range(CALENDAR_START, CALENDAR_END, freq="D")
    is_bank_holiday = dates.isin(BANK_HOLIDAYS)
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "month_name": dates.strftime("%b"),
        "year_month": dates.strftime("%Y-%m"),
        "weekday_name": dates.day_name(),
        "is_bank_holiday": is_bank_holiday,
        "day_type": np.where((dates.dayofweek >= 5) | is_bank_holiday, "Weekend/Holiday", "Weekday"),
    })


def off_peak_mask(tariff_type):
    """Boolean array over the 48 half-hours (hh_0 = 00:00-00:30), True when off-peak."""
    starts = np.arange(48) * 30
    mask = np.zeros(48, dtype=bool)
    for start, end in TOU_TARIFFS[tariff_type]["off_peak_windows"]:
        start_min, end_min = (int(t[:2]) * 60 + int(t[3:]) for t in (start, end))
        mask |= (starts >= start_min) & (starts < end_min)
    return mask


def build_tariffs(households):
    """Dummy tariff per household (seeded, reproducible).

    Std households get a single unit rate; ToU households get Economy 7 or
    Economy 10 peak/off-peak rates. Everyone has a standing charge.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(households)
    unit_rate = rng.uniform(*UNIT_RATE_RANGE_P_PER_KWH, n).round(2)
    standing_charge = rng.uniform(*STANDING_CHARGE_RANGE_P_PER_DAY, n).round(2)

    # ToU draws come after the original draws so Std tariffs are unchanged
    is_tou = households["stdorToU"].eq("ToU").to_numpy()
    tariff_type = np.where(is_tou, rng.choice(list(TOU_TARIFFS), n), "Standard")
    peak_rate = np.full(n, np.nan)
    off_peak_rate = np.full(n, np.nan)
    off_peak_hours = np.full(n, "", dtype=object)
    for name, spec in TOU_TARIFFS.items():
        selected = tariff_type == name
        peak_rate[selected] = rng.uniform(*spec["peak_range"], n).round(2)[selected]
        off_peak_rate[selected] = rng.uniform(*spec["off_peak_range"], n).round(2)[selected]
        off_peak_hours[selected] = ", ".join(f"{s}-{e}" for s, e in spec["off_peak_windows"])

    return pd.DataFrame({
        "LCLid": households["LCLid"].to_numpy(),
        "tariff_type": tariff_type,
        "variable_rate_p_per_kwh": np.where(is_tou, np.nan, unit_rate),
        "peak_rate_p_per_kwh": peak_rate,
        "off_peak_rate_p_per_kwh": off_peak_rate,
        "off_peak_hours": off_peak_hours,
        "daily_charge_p_per_day": standing_charge,
        "currency": "GBP",
    })


def load_off_peak_history(households, tariffs):
    """Daily total and off-peak kWh from the half-hourly blocks, for ToU-tariff households."""
    tou = tariffs.loc[tariffs["tariff_type"] != "Standard", ["LCLid", "tariff_type"]]
    blocks = households.loc[households["LCLid"].isin(tou["LCLid"]), "block"].unique()
    hh_cols = [f"hh_{i}" for i in range(48)]
    masks = {name: off_peak_mask(name) for name in TOU_TARIFFS}

    frames = []
    for block in sorted(blocks):
        hh = pd.read_csv(HHBLOCK_DIR / f"{block}.csv", dtype={"LCLid": "string"}, parse_dates=["day"])
        hh = hh.merge(tou, on="LCLid")
        values = hh[hh_cols].to_numpy(dtype=float)
        mask = np.stack([masks[t] for t in hh["tariff_type"]])
        frames.append(pd.DataFrame({
            "LCLid": hh["LCLid"],
            "date": hh["day"],
            "hh_kwh": np.nansum(values, axis=1),
            "off_peak_hh_kwh": np.nansum(np.where(mask, values, 0.0), axis=1),
        }))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["LCLid", "date"])


def split_off_peak(frame, off_peak_hist, calendar, tariffs, train_end=None):
    """Add peak_kwh / off_peak_kwh to a daily kWh frame (left blank for Standard tariffs).

    With train_end=None (actuals) each day's own half-hourly split is used where known.
    Otherwise, and as a fallback, the household's off-peak share by month x day type
    from history up to train_end is applied.
    """
    hist = off_peak_hist if train_end is None else off_peak_hist[off_peak_hist["date"] <= train_end]
    hist = hist.merge(calendar[["date", "month", "day_type"]], on="date")
    hist = hist.merge(tariffs[["LCLid", "tariff_type"]], on="LCLid")

    def share(keys):
        sums = hist.groupby(keys, as_index=False)[["off_peak_hh_kwh", "hh_kwh"]].sum()
        sums["share"] = sums["off_peak_hh_kwh"] / sums["hh_kwh"].where(sums["hh_kwh"] > 0)
        return sums[keys + ["share"]]

    frame = (
        frame.merge(tariffs[["LCLid", "tariff_type"]], on="LCLid")
        .merge(calendar[["date", "month", "day_type"]], on="date")
    )
    tou = frame["tariff_type"] != "Standard"
    rows = frame.loc[tou, ["LCLid", "date", "month", "day_type", "tariff_type"]]
    if train_end is None:
        day = hist[["LCLid", "date"]].assign(share=hist["off_peak_hh_kwh"] / hist["hh_kwh"].where(hist["hh_kwh"] > 0))
        rows = rows.merge(day, how="left", on=["LCLid", "date"])
    else:
        rows = rows.assign(share=np.nan)
    for keys in (["LCLid", "month", "day_type"], ["LCLid"], ["tariff_type"]):
        fallback = rows[keys].merge(share(keys), how="left", on=keys)["share"].to_numpy()
        rows["share"] = rows["share"].fillna(pd.Series(fallback, index=rows.index))

    frame["off_peak_kwh"] = np.nan
    frame.loc[tou, "off_peak_kwh"] = frame.loc[tou, "kwh"].to_numpy() * rows["share"].to_numpy()
    frame["peak_kwh"] = frame["kwh"] - frame["off_peak_kwh"]
    return frame[["LCLid", "date", "kwh", "peak_kwh", "off_peak_kwh"]]


def seasonal_index(hist):
    """Monthly index (month mean / average month) per Acorn group, used to fill gaps."""
    overall = hist.groupby("month")["kwh"].mean()
    overall = overall / overall.mean()

    monthly = hist.groupby(["Acorn_grouped", "month"])["kwh"].mean()
    cohort = monthly / monthly.groupby(level="Acorn_grouped").transform("mean")
    cohort_size = hist.groupby("Acorn_grouped")["LCLid"].nunique()

    groups = hist["Acorn_grouped"].unique()
    index = pd.MultiIndex.from_product([groups, range(1, 13)], names=["Acorn_grouped", "month"])
    result = cohort.reindex(index).rename("seasonal_index").reset_index()
    small = result["Acorn_grouped"].map(cohort_size) < MIN_COHORT_HOUSEHOLDS
    result.loc[small, "seasonal_index"] = np.nan
    result["seasonal_index"] = result["seasonal_index"].fillna(result["month"].map(overall))
    return result


def forecast(daily, calendar, households, train_end, horizon_end):
    """Daily kWh forecast for (train_end, horizon_end] for households active at train_end."""
    hist = daily[daily["date"] <= train_end]
    active_from = train_end - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)
    active = hist.loc[hist["date"] > active_from, "LCLid"].unique()

    hist = (
        hist[hist["LCLid"].isin(active)]
        .merge(calendar[["date", "month", "day_type"]], on="date")
        .merge(households[["LCLid", "Acorn_grouped"]], on="LCLid")
    )
    index = seasonal_index(hist)
    hist = hist.merge(index, on=["Acorn_grouped", "month"])

    profile = hist.groupby(["LCLid", "month", "day_type"], as_index=False)["kwh"].mean()
    profile = profile.rename(columns={"kwh": "profile_kwh"})
    hist["deseasonalised_kwh"] = hist["kwh"] / hist["seasonal_index"]
    base_by_type = hist.groupby(["LCLid", "day_type"], as_index=False)["deseasonalised_kwh"].mean()
    base_by_type = base_by_type.rename(columns={"deseasonalised_kwh": "base_kwh"})
    base_overall = hist.groupby("LCLid", as_index=False)["deseasonalised_kwh"].mean()
    base_overall = base_overall.rename(columns={"deseasonalised_kwh": "base_overall_kwh"})

    def expected_kwh(frame):
        # Household profile where it exists, otherwise household base level x seasonal index
        frame = (
            frame[["LCLid", "month", "day_type", "seasonal_index"]]
            .merge(profile, how="left", on=["LCLid", "month", "day_type"])
            .merge(base_by_type, how="left", on=["LCLid", "day_type"])
            .merge(base_overall, how="left", on="LCLid")
        )
        fallback = frame["base_kwh"].fillna(frame["base_overall_kwh"]) * frame["seasonal_index"]
        return frame["profile_kwh"].fillna(fallback).to_numpy()

    recent = hist[hist["date"] > train_end - pd.Timedelta(days=LEVEL_WINDOW_DAYS)].copy()
    recent["expected_kwh"] = expected_kwh(recent)
    level = recent.groupby("LCLid")[["kwh", "expected_kwh"]].sum()
    level = (level["kwh"] / level["expected_kwh"]).replace([np.inf, -np.inf], np.nan)
    level = level.fillna(1.0).clip(*LEVEL_CLIP).rename("level_factor").reset_index()

    horizon = calendar.loc[
        (calendar["date"] > train_end) & (calendar["date"] <= horizon_end),
        ["date", "month", "day_type"],
    ]
    grid = (
        pd.DataFrame({"LCLid": active})
        .merge(horizon, how="cross")
        .merge(households[["LCLid", "Acorn_grouped"]], on="LCLid")
        .merge(index, how="left", on=["Acorn_grouped", "month"])
        .merge(level, on="LCLid")
    )
    grid["kwh"] = expected_kwh(grid) * grid["level_factor"]
    return grid[["LCLid", "date", "kwh"]]


def add_costs(fact, tariffs):
    rate_cols = ["variable_rate_p_per_kwh", "peak_rate_p_per_kwh", "off_peak_rate_p_per_kwh", "daily_charge_p_per_day"]
    fact = fact.merge(tariffs[["LCLid", "tariff_type"] + rate_cols], on="LCLid")
    two_rate = fact["peak_kwh"] * fact["peak_rate_p_per_kwh"] + fact["off_peak_kwh"] * fact["off_peak_rate_p_per_kwh"]
    single_rate = fact["kwh"] * fact["variable_rate_p_per_kwh"]
    fact["variable_cost_gbp"] = np.where(fact["tariff_type"] == "Standard", single_rate, two_rate) / 100
    fact["fixed_cost_gbp"] = fact["daily_charge_p_per_day"] / 100
    fact["total_cost_gbp"] = fact["variable_cost_gbp"] + fact["fixed_cost_gbp"]
    return fact.drop(columns=["tariff_type"] + rate_cols)


def backtest_accuracy(backtest, actual):
    """Compare backtest forecast with actuals (both already costed) on days where both exist."""
    matched = backtest.merge(actual, on=["LCLid", "date"], suffixes=("_forecast", "_actual"))
    for kind in ("forecast", "actual"):
        matched[f"{kind}_cost_gbp"] = matched[f"total_cost_gbp_{kind}"]
    matched["abs_error_kwh"] = (matched["kwh_forecast"] - matched["kwh_actual"]).abs()

    per_household = matched.groupby("LCLid").agg(
        days=("date", "size"),
        actual_kwh=("kwh_actual", "sum"),
        forecast_kwh=("kwh_forecast", "sum"),
        abs_error_kwh=("abs_error_kwh", "sum"),
        actual_cost_gbp=("actual_cost_gbp", "sum"),
        forecast_cost_gbp=("forecast_cost_gbp", "sum"),
    )
    per_household["wape"] = per_household["abs_error_kwh"] / per_household["actual_kwh"]
    per_household["bias_pct"] = per_household["forecast_kwh"] / per_household["actual_kwh"] - 1
    per_household = per_household.drop(columns="abs_error_kwh").replace([np.inf, -np.inf], np.nan)

    monthly = matched.groupby(["LCLid", matched["date"].dt.to_period("M")])[["kwh_forecast", "kwh_actual"]].sum()
    summary = {
        "households": matched["LCLid"].nunique(),
        "daily WAPE": matched["abs_error_kwh"].sum() / matched["kwh_actual"].sum(),
        "monthly WAPE": (monthly["kwh_forecast"] - monthly["kwh_actual"]).abs().sum() / monthly["kwh_actual"].sum(),
        "bias": matched["kwh_forecast"].sum() / matched["kwh_actual"].sum() - 1,
        "MAE kWh/day": matched["abs_error_kwh"].mean(),
        "actual cost GBP": matched["actual_cost_gbp"].sum(),
        "forecast cost GBP": matched["forecast_cost_gbp"].sum(),
        "ToU off-peak share, actual": matched["off_peak_kwh_actual"].sum()
        / matched["kwh_actual"].where(matched["off_peak_kwh_actual"].notna()).sum(),
        "ToU off-peak share, forecast": matched["off_peak_kwh_forecast"].sum()
        / matched["kwh_forecast"].where(matched["off_peak_kwh_forecast"].notna()).sum(),
    }
    backtest_cols = list(backtest.columns)
    matched_backtest = matched.rename(columns={f"{c}_forecast": c for c in backtest_cols})[backtest_cols]
    return matched_backtest, per_household.reset_index(), summary


def sanity_checks(daily, final_forecast):
    print("\nSanity checks")
    print(f"  NaN forecast kWh: {final_forecast['kwh'].isna().sum()}")
    print(f"  Negative forecast kWh: {(final_forecast['kwh'] < 0).sum()}")
    days = final_forecast.groupby("LCLid").size()
    print(f"  Forecast days per household: min {days.min()}, max {days.max()}")

    ids = final_forecast["LCLid"].unique()
    last_year = daily[
        daily["LCLid"].isin(ids) & (daily["date"] > FORECAST_TRAIN_END - pd.DateOffset(years=1))
    ]
    actual_mean = last_year["kwh"].mean()
    forecast_mean = final_forecast["kwh"].mean()
    print(f"  Mean daily kWh, actual Mar13-Feb14: {actual_mean:.2f}")
    print(f"  Mean daily kWh, forecast Mar14-Feb15: {forecast_mean:.2f} ({forecast_mean / actual_mean - 1:+.1%})")


def spot_check(fact, lclid):
    rows = fact[fact["LCLid"] == lclid]
    table = rows.pivot_table(
        index=rows["date"].dt.strftime("%Y-%m"), columns="scenario", values="kwh", aggfunc="sum"
    )
    print(f"\nSpot check {lclid}: monthly kWh")
    print(table.round(1).fillna("").to_string())


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    daily = load_daily()
    households = load_households()
    calendar = build_calendar()
    tariffs = build_tariffs(households)
    print(f"  {len(daily):,} clean household-days for {daily['LCLid'].nunique():,} households")
    print("  Tariffs: " + ", ".join(f"{k} {v:,}" for k, v in tariffs["tariff_type"].value_counts().items()))
    print("Loading half-hourly data for ToU households...")
    off_peak_hist = load_off_peak_history(households, tariffs)
    actual = add_costs(split_off_peak(daily, off_peak_hist, calendar, tariffs), tariffs)

    print("Backtest: train to 2013-11-30, forecast Dec 2013 - Feb 2014...")
    backtest = forecast(daily, calendar, households, BACKTEST_TRAIN_END, BACKTEST_END)
    backtest = add_costs(split_off_peak(backtest, off_peak_hist, calendar, tariffs, BACKTEST_TRAIN_END), tariffs)
    backtest, accuracy, summary = backtest_accuracy(backtest, actual)
    for name, value in summary.items():
        if "WAPE" in name or name == "bias" or "share" in name:
            print(f"  {name}: {value:.1%}")
        elif "GBP" in name:
            print(f"  {name}: {value:,.0f}")
        else:
            print(f"  {name}: {value:,.2f}" if isinstance(value, float) else f"  {name}: {value:,}")

    print("Forecast: train to 2014-02-28, forecast Mar 2014 - Feb 2015...")
    final_forecast = forecast(daily, calendar, households, FORECAST_TRAIN_END, FORECAST_END)
    print(f"  {final_forecast['LCLid'].nunique():,} households forecast")
    sanity_checks(daily, final_forecast)
    final_forecast = add_costs(
        split_off_peak(final_forecast, off_peak_hist, calendar, tariffs, FORECAST_TRAIN_END), tariffs
    )

    fact = pd.concat([
        actual.assign(scenario="Actual"),
        backtest.assign(scenario="Backtest"),
        final_forecast.assign(scenario="Forecast"),
    ], ignore_index=True)
    value_cols = ["kwh", "peak_kwh", "off_peak_kwh", "variable_cost_gbp", "fixed_cost_gbp", "total_cost_gbp"]
    fact = fact[["LCLid", "date", "scenario"] + value_cols].round(dict.fromkeys(value_cols, 4))

    spot_id = "MAC000002" if "MAC000002" in set(final_forecast["LCLid"]) else final_forecast["LCLid"].iloc[0]
    spot_check(fact, spot_id)

    span = daily.groupby("LCLid")["date"].agg(first_date="min", last_date="max").reset_index()
    dim_household = households.merge(span, how="left", on="LCLid")
    dim_household["in_forecast"] = dim_household["LCLid"].isin(final_forecast["LCLid"])

    outputs = {
        "dim_household.csv": dim_household,
        "dim_tariff.csv": tariffs,
        "dim_date.csv": calendar,
        "fact_daily_energy.csv": fact,
        "backtest_accuracy.csv": accuracy.round(4),
    }
    print("\nWriting outputs...")
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / name, index=False, date_format="%Y-%m-%d")
        print(f"  {name}: {len(frame):,} rows")
    print("\nRows by scenario:")
    print(fact["scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
