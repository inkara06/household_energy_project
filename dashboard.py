"""Streamlit mini-dashboard.

Run:  streamlit run dashboard.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, forecast, pipeline, preprocessing, timeseries

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
DAY_TYPES = ["weekday", "weekend"]

st.set_page_config(page_title="Household Energy", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = preprocessing.add_calendar_features(pipeline.load_hourly())
    daily = preprocessing.add_calendar_features(preprocessing.to_daily(pipeline.load_hourly()))
    return hourly, daily


@st.cache_data
def load_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, daily = load_data()
    return forecast.compare_models(daily["energy_kwh"].astype("float64"))


hourly, daily = load_data()

st.sidebar.header("Filters")
years = sorted(daily["year"].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)
selected_seasons = st.sidebar.multiselect("Season", SEASON_ORDER, default=SEASON_ORDER)
selected_day_types = st.sidebar.multiselect("Day type", DAY_TYPES, default=DAY_TYPES)
meter_names = {code: name for code, (name, _) in config.SUB_METERS.items()}
selected_meters = st.sidebar.multiselect("Category (sub-meter)", list(meter_names),
                                         default=list(meter_names), format_func=meter_names.get)


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    day_type = frame["is_weekend"].map({False: "weekday", True: "weekend"})
    mask = (frame["year"].isin(selected_years)
            & frame["season"].isin(selected_seasons)
            & day_type.isin(selected_day_types))
    return frame[mask]


hourly_f = apply_filters(hourly)
daily_f = apply_filters(daily)

st.title("Household Energy Consumption — Sceaux, France (2006–2010)")
if daily_f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

kpi = st.columns(4)
kpi[0].metric("Days selected", f"{len(daily_f):,}")
kpi[1].metric("Average per day", f"{daily_f['energy_kwh'].mean():.1f} kWh")
kpi[2].metric("Max day", f"{daily_f['energy_kwh'].max():.1f} kWh")
kpi[3].metric("Peak hour", f"{hourly_f['energy_kwh'].max():.2f} kWh")

tab_trend, tab_profiles, tab_meters, tab_forecast = st.tabs(
    ["Trend", "Load profiles", "Sub-meters", "Forecast"])

with tab_trend:
    window = st.slider("Moving average window, days", 1, 60, 7)
    trend = daily_f[["energy_kwh"]].copy()
    trend["moving_average"] = trend["energy_kwh"].rolling(window, min_periods=1).mean()
    st.plotly_chart(px.line(trend, labels={"value": "kWh per day", "datetime": "Date"},
                            title="Daily consumption"), width="stretch")

with tab_profiles:
    left, right = st.columns(2)
    profile = hourly_f.groupby(["hour", "season"])["energy_kwh"].mean().reset_index()
    left.plotly_chart(px.line(profile, x="hour", y="energy_kwh", color="season",
                              category_orders={"season": SEASON_ORDER},
                              labels={"energy_kwh": "Average kWh per hour", "hour": "Hour of day"},
                              title="Daily load profile by season"), width="stretch")
    weekday = daily_f.groupby("weekday")["energy_kwh"].mean().reset_index()
    weekday["weekday"] = weekday["weekday"].map(dict(enumerate(timeseries.WEEKDAY_NAMES)))
    right.plotly_chart(px.bar(weekday, x="weekday", y="energy_kwh",
                              labels={"energy_kwh": "Average kWh per day", "weekday": ""},
                              title="Weekly profile"), width="stretch")
    heat = hourly_f.groupby(["hour", "month"])["energy_kwh"].mean().unstack()
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                              labels={"x": "Month", "y": "Hour", "color": "kWh"},
                              title="Average load: hour × month"), width="stretch")

with tab_meters:
    if not selected_meters:
        st.info("Select at least one sub-meter in the sidebar.")
    else:
        monthly = daily_f[selected_meters].resample("ME").sum() / config.WATTS_PER_KILOWATT
        monthly = monthly.rename(columns=meter_names)
        st.plotly_chart(px.bar(monthly, labels={"value": "kWh", "datetime": "Month", "variable": "Sub-meter"},
                               title="Monthly energy by sub-meter"), width="stretch")
        shares = monthly.sum()
        st.plotly_chart(px.pie(values=shares.values, names=shares.index, title="Share of energy"),
                        width="stretch")

with tab_forecast:
    metrics, predictions = load_forecasts()
    st.subheader("Model comparison on 2010 hold-out (day-ahead)")
    st.dataframe(metrics, hide_index=True)
    models = st.multiselect("Models to plot", list(predictions.columns.drop("actual")),
                            default=["Naive (baseline)", "Linear regression (lags)"])
    st.plotly_chart(px.line(predictions[["actual"] + models],
                            labels={"value": "kWh per day", "datetime": "Date"},
                            title="Forecast vs actual"), width="stretch")
    horizon = st.number_input("Forecast next N days", min_value=1, max_value=30, value=7)
    future = forecast.forecast_future(daily["energy_kwh"].astype("float64"), int(horizon))
    st.plotly_chart(px.bar(future, labels={"value": "kWh per day", "index": "Date"},
                           title="Future forecast (linear regression, recursive)"),
                    width="stretch")
