# Energy Forecaster

Forecasts daily electricity use and spend for each household (`LCLid`) in the Kaggle
[Smart meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) dataset,
and writes the results as CSVs ready for a Power BI model.

## Run

```
pip install -r requirements.txt
python energy_forecast.py
```

Input: `data/archive/` (the extracted Kaggle download). Output: `output/`. The run takes about two minutes.

## Method

A seasonal profile forecast with a recent-level adjustment. It uses no machine learning.

1. **Clean.** Daily totals come from `daily_dataset.csv`. Days with fewer than 46 of 48 half-hour readings are dropped. Days with 46–47 readings are scaled up to 48.
2. **Profile.** For each household, the mean daily kWh by calendar month × day type. The day types are *Weekday* and *Weekend/Holiday*, using England & Wales bank holidays.
3. **Level factor.** Actual kWh over the last 8 weeks of training ÷ what the profile expected for those weeks. It is clipped to 0.5–2.0.
4. **Forecast.** Daily kWh = profile × level factor. If a household has no history for a month, its base level is multiplied by the seasonal index of its Acorn group.
5. **Peak/off-peak split (ToU households only).** Each day's kWh is split into peak and off-peak.
   - For actuals, the split comes from that day's half-hourly readings (`hhblock_dataset`).
   - For forecasts, it comes from the household's historical off-peak share by month × day type.
6. **Spend.**
   - Variable cost:
     - Standard tariff: kWh × unit rate.
     - Economy 7 / 10: peak kWh × peak rate + off-peak kWh × off-peak rate.
     - In both cases, rates are in p/kWh and the result is divided by 100 to give £.
   - Fixed cost = standing charge (p/day) / 100.
   - Total cost = variable cost + fixed cost.

| Run | Train to | Forecast period |
|---|---|---|
| Backtest | 2013-11-30 | Dec 2013 – Feb 2014 (compared to actuals) |
| Forecast | 2014-02-28 | Mar 2014 – Feb 2015 |

Only households with data in the last 28 days before the training cut-off are forecast. About 500 households left the trial early, so they appear in the actuals but are not forecast.

**Dummy tariffs.** Every household gets a random standing charge of 40–65 p/day. The unit rates depend on the household's `stdorToU` value in `informations_households.csv`:

| Households | Tariff | Unit rates (p/kWh) | Off-peak hours |
|---|---|---|---|
| Std (4,443) | Standard | 20–35 single rate | n/a |
| ToU, about half (561) | Economy 7 | Peak 28–38, off-peak 12–18 | 00:30–07:30 |
| ToU, about half (562) | Economy 10 | Peak 28–36, off-peak 16–22 | 00:00–05:00, 13:00–16:00, 20:00–22:00 |

The random seed is 42, so reruns give the same tariffs. To change the ranges or off-peak windows, edit the constants at the top of `energy_forecast.py`. Off-peak windows use local clock time all year. Real Economy 7 meters often stay on GMT, which shifts the window an hour later in summer; that is ignored here.

**Bank holidays.** The supplied `uk_bank_holidays.csv` has day/month swaps and no 2015 dates, so a corrected list is hard-coded in the script.

## Outputs (star schema)

| File | Grain | Notes |
|---|---|---|
| `dim_household.csv` | LCLid | Tariff type (Std/ToU), Acorn group, first/last date, `in_forecast` flag |
| `dim_tariff.csv` | LCLid | `tariff_type` (Standard / Economy 7 / Economy 10). Std households have `variable_rate_p_per_kwh`; ToU households have `peak_rate_p_per_kwh`, `off_peak_rate_p_per_kwh` and `off_peak_hours`. All have `daily_charge_p_per_day` and currency. |
| `dim_date.csv` | date | Year, month, `year_month`, weekday, `day_type`, `is_bank_holiday` |
| `fact_daily_energy.csv` | LCLid × date × scenario | `scenario` = Actual / Backtest / Forecast. Columns: `kwh`, `peak_kwh`, `off_peak_kwh` (blank for Standard tariffs), and costs in GBP. |
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
| Forecast spend vs actual | £1.74m vs £1.67m |
| ToU off-peak share, forecast vs actual | 26.2% vs 26.3% |

**Limitation.** The model doesn't use weather. Unusual months carry into the forecast: for example, the very cold March 2013 pushes up the March 2014 forecast. Adding weather (`weather_daily_darksky.csv`) is a natural next step.
