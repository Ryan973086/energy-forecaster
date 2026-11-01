# Energy Forecaster

Forecasts daily electricity use and spend for each household (`LCLid`) in the Kaggle
[Smart meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) dataset,
and writes the results as CSVs ready for a Power BI model.

## Run

```
pip install -r requirements.txt
python energy_forecast.py
```

Input: `data/archive/` (the extracted Kaggle download). Output: `output/`. The run takes about one minute.

## Method

A seasonal profile forecast with a recent-level adjustment. It uses no machine learning.

1. **Clean.** Daily totals come from `daily_dataset.csv`. Days with fewer than 46 of 48 half-hour readings are dropped. Days with 46–47 readings are scaled up to 48.
2. **Profile.** For each household, the mean daily kWh by calendar month × day type. The day types are *Weekday* and *Weekend/Holiday*, using England & Wales bank holidays.
3. **Level factor.** Actual kWh over the last 8 weeks of training ÷ what the profile expected for those weeks. It is clipped to 0.5–2.0.
4. **Forecast.** Daily kWh = profile × level factor. If a household has no history for a month, its base level is multiplied by the seasonal index of its Acorn group.
5. **Spend.**
   - Variable cost = kWh × unit rate (p/kWh) / 100.
   - Fixed cost = standing charge (p/day) / 100.
   - Total cost = variable cost + fixed cost.

| Run | Train to | Forecast period |
|---|---|---|
| Backtest | 2013-11-30 | Dec 2013 – Feb 2014 (compared to actuals) |
| Forecast | 2014-02-28 | Mar 2014 – Feb 2015 |

Only households with data in the last 28 days before the training cut-off are forecast. About 500 households left the trial early, so they appear in the actuals but are not forecast.

**Dummy tariffs.** Each household gets a random unit rate of 20–35 p/kWh and a standing charge of 40–65 p/day. The random seed is 42, so reruns give the same tariffs. To change the ranges, edit the constants at the top of `energy_forecast.py`.

**Bank holidays.** The supplied `uk_bank_holidays.csv` has day/month swaps and no 2015 dates, so a corrected list is hard-coded in the script.

## Outputs (star schema)

| File | Grain | Notes |
|---|---|---|
| `dim_household.csv` | LCLid | Tariff type (Std/ToU), Acorn group, first/last date, `in_forecast` flag |
| `dim_tariff.csv` | LCLid | `variable_rate_p_per_kwh`, `daily_charge_p_per_day`, currency |
| `dim_date.csv` | date | Year, month, `year_month`, weekday, `day_type`, `is_bank_holiday` |
| `fact_daily_energy.csv` | LCLid × date × scenario | `scenario` = Actual / Backtest / Forecast; kWh and costs in GBP |
| `backtest_accuracy.csv` | LCLid | Backtest actual vs forecast kWh and cost, WAPE, bias |

Backtest rows are kept only on days that also have an actual reading, so Actual and Backtest totals compare like for like.

**Power BI relationships.** Each dimension joins one-to-many to `fact_daily_energy`:
- `dim_household[LCLid]`
- `dim_tariff[LCLid]`
- `dim_date[date]`

Filter or slice on `scenario`.

## Backtest results (Dec 2013 – Feb 2014, 5,134 households)

| Metric | Value |
|---|---|
| Daily WAPE | 27.5% |
| Monthly WAPE (household × month) | 17.9% |
| Overall bias | +5.0% |
| Forecast spend vs actual | £1.73m vs £1.66m |

**Limitation.** The model doesn't use weather. Unusual months carry into the forecast: for example, the very cold March 2013 pushes up the March 2014 forecast. Adding weather (`weather_daily_darksky.csv`) is a natural next step.
