"""Forecast household electricity spend for the London smart meter dataset.

Method (deliberately simple, no ML):
  1. Seasonal profile: each household's mean daily kWh by calendar month x day type.
  2. Level factor: recent 8 weeks of actuals / what the profile expected for those
     weeks, so the forecast follows any recent change in usage.
  3. Forecast daily kWh = profile x level factor.
Spend = kWh x dummy unit rate (p/kWh) + dummy standing charge (p/day).

Outputs a Power BI-ready star schema of CSVs in ./output.
Run: python energy_forecast.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "archive"
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


def build_tariffs(households):
    """Dummy unit rate and standing charge per household (seeded, reproducible)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(households)
    return pd.DataFrame({
        "LCLid": households["LCLid"].to_numpy(),
        "variable_rate_p_per_kwh": rng.uniform(*UNIT_RATE_RANGE_P_PER_KWH, n).round(2),
        "daily_charge_p_per_day": rng.uniform(*STANDING_CHARGE_RANGE_P_PER_DAY, n).round(2),
        "currency": "GBP",
    })


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
    fact = fact.merge(tariffs[["LCLid", "variable_rate_p_per_kwh", "daily_charge_p_per_day"]], on="LCLid")
    fact["variable_cost_gbp"] = fact["kwh"] * fact["variable_rate_p_per_kwh"] / 100
    fact["fixed_cost_gbp"] = fact["daily_charge_p_per_day"] / 100
    fact["total_cost_gbp"] = fact["variable_cost_gbp"] + fact["fixed_cost_gbp"]
    return fact.drop(columns=["variable_rate_p_per_kwh", "daily_charge_p_per_day"])


def backtest_accuracy(backtest, daily, tariffs):
    """Compare backtest forecast with actuals on days where both exist."""
    matched = backtest.merge(daily, on=["LCLid", "date"], suffixes=("_forecast", "_actual"))
    matched = matched.merge(tariffs, on="LCLid")
    for kind in ("forecast", "actual"):
        matched[f"{kind}_cost_gbp"] = (
            matched[f"kwh_{kind}"] * matched["variable_rate_p_per_kwh"] + matched["daily_charge_p_per_day"]
        ) / 100
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
    }
    return matched[["LCLid", "date", "kwh_forecast"]].rename(columns={"kwh_forecast": "kwh"}), \
        per_household.reset_index(), summary


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

    print("Backtest: train to 2013-11-30, forecast Dec 2013 - Feb 2014...")
    backtest = forecast(daily, calendar, households, BACKTEST_TRAIN_END, BACKTEST_END)
    backtest, accuracy, summary = backtest_accuracy(backtest, daily, tariffs)
    for name, value in summary.items():
        if "WAPE" in name or name == "bias":
            print(f"  {name}: {value:.1%}")
        elif "GBP" in name:
            print(f"  {name}: {value:,.0f}")
        else:
            print(f"  {name}: {value:,.2f}" if isinstance(value, float) else f"  {name}: {value:,}")

    print("Forecast: train to 2014-02-28, forecast Mar 2014 - Feb 2015...")
    final_forecast = forecast(daily, calendar, households, FORECAST_TRAIN_END, FORECAST_END)
    print(f"  {final_forecast['LCLid'].nunique():,} households forecast")
    sanity_checks(daily, final_forecast)

    fact = pd.concat([
        daily.assign(scenario="Actual"),
        backtest.assign(scenario="Backtest"),
        final_forecast.assign(scenario="Forecast"),
    ], ignore_index=True)
    fact = add_costs(fact, tariffs)
    fact = fact[["LCLid", "date", "scenario", "kwh", "variable_cost_gbp", "fixed_cost_gbp", "total_cost_gbp"]]
    fact = fact.round({"kwh": 4, "variable_cost_gbp": 4, "fixed_cost_gbp": 4, "total_cost_gbp": 4})

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
