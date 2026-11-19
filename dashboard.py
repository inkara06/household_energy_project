"""Streamlit dashboard: load clock, trends, profiles, sub-meters and forecasts.

Run:  streamlit run dashboard.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

from src import config, forecast, pipeline, preprocessing, timeseries

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
DAY_TYPES = ["weekday", "weekend"]
HOURS_PER_DAY = config.HOURS_PER_DAY
DEGREES_PER_HOUR = 360 / HOURS_PER_DAY

INK = "#0D1B36"
PANEL = "#14264B"
TEXT = "#E6ECF7"
MUTED = "#8E9CB8"
GRID = "rgba(142, 156, 184, 0.16)"
AMBER = "#F2A33A"
GLOW = "#FFD58A"
STEEL = "#7FA2E0"
SEASON_COLORS = {"winter": "#6E9BF2", "spring": "#5FC6A2", "summer": "#F2C94C", "autumn": "#E0713F"}
METER_COLORS = {"kitchen": "#5FC6A2", "laundry": "#B08CF2", "water_heater_ac": AMBER, "other": "#4A6190"}
MODEL_COLORS = {
    "actual": TEXT,
    "Naive (baseline)": "#8E9CB8",
    "Seasonal naive": "#B08CF2",
    "Moving average (7)": "#5FC6A2",
    "Linear regression (lags)": AMBER,
}
HEAT_SCALE = [[0.0, INK], [0.35, "#2A3F78"], [0.7, AMBER], [1.0, GLOW]]
FONT_BODY = "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
FONT_DISPLAY = "'Unbounded', 'Arial Black', 'Helvetica Neue', sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Unbounded:wght@500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, p {{ font-family: {FONT_BODY}; }}
.stApp {{
    background:
        radial-gradient(1200px 520px at 78% -8%, rgba(242, 163, 58, 0.16), transparent 62%),
        radial-gradient(900px 480px at -10% 12%, rgba(110, 155, 242, 0.12), transparent 60%),
        {INK};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #0A1630; border-right: 1px solid rgba(142,156,184,.14); }}
[data-testid="stSidebar"] h2 {{ font-family: {FONT_DISPLAY}; font-size: 1.05rem; letter-spacing: .01em; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
.hero-title {{
    font-family: {FONT_DISPLAY}; font-weight: 700; line-height: 1.04;
    font-size: clamp(2rem, 3.6vw, 3rem); margin: .2rem 0 1rem 0; color: {TEXT};
}}
.hero-title span {{ display: block; color: {MUTED}; font-family: {FONT_BODY}; font-weight: 500;
    font-size: .4em; line-height: 1.3; margin-top: .7rem; }}
.hero-lede {{ font-size: 1.08rem; line-height: 1.6; color: #C9D3E8; max-width: 34rem; }}
.hero-lede b {{ color: {GLOW}; font-weight: 600; }}
.kpi {{ border-top: 2px solid rgba(242,163,58,.55); padding-top: .7rem; }}
.kpi-value {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 2rem; color: {TEXT}; line-height: 1.1; }}
.kpi-value small {{ font-family: {FONT_BODY}; font-size: .9rem; color: {MUTED}; font-weight: 500; margin-left: .3rem; }}
.kpi-label {{ color: {MUTED}; font-size: .92rem; margin-top: .25rem; }}
.section-note {{ color: {MUTED}; font-size: .95rem; margin: -.4rem 0 .6rem 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid rgba(142,156,184,.18); }}
.stTabs [data-baseweb="tab"] {{ font-family: {FONT_DISPLAY}; font-size: .95rem; padding: .6rem 0; }}
.stTabs [aria-selected="true"] {{ color: {AMBER}; }}
</style>
"""


def build_template() -> go.layout.Template:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "color": TEXT, "size": 13},
        title={"font": {"family": FONT_DISPLAY, "size": 16, "color": TEXT}, "x": 0, "xanchor": "left"},
        colorway=[AMBER, STEEL, "#5FC6A2", "#B08CF2", "#E0713F", "#F2C94C"],
        xaxis={"gridcolor": GRID, "zeroline": False, "linecolor": GRID, "tickfont": {"color": MUTED}, "automargin": True,
               "title": {"font": {"color": MUTED}}},
        yaxis={"gridcolor": GRID, "zeroline": False, "linecolor": GRID, "tickfont": {"color": MUTED}, "automargin": True,
               "title": {"font": {"color": MUTED}}},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}, "title": {"text": ""}},
        hoverlabel={"bgcolor": PANEL, "bordercolor": AMBER, "font": {"family": FONT_BODY, "color": TEXT}},
        margin={"l": 56, "r": 16, "t": 64, "b": 48},
    )
    return template


pio.templates["sceaux"] = build_template()
pio.templates.default = "sceaux"

st.set_page_config(page_title="Sceaux household energy", page_icon="⚡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = preprocessing.add_calendar_features(pipeline.load_hourly())
    daily = preprocessing.add_calendar_features(preprocessing.to_daily(pipeline.load_hourly()))
    return hourly, daily


@st.cache_data
def load_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, daily = load_data()
    return forecast.compare_models(daily["energy_kwh"].astype("float64"))


def show(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", theme=None, config={"displayModeBar": False})


CLOCK_STYLES = {"weekday": (STEEL, "rgba(127,162,224,0.22)"), "weekend": (AMBER, "rgba(242,163,58,0.30)")}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
CLOCK_NOTE = "distance from centre = average kWh in that hour"


def day_type_of(frame: pd.DataFrame) -> pd.Series:
    return frame["is_weekend"].map({False: "weekday", True: "weekend"})


def clock_trace(values: pd.Series, kind: str) -> go.Scatterpolar:
    """One closed 24-hour loop; values are indexed by hour 0..23."""
    closed = pd.concat([values.reindex(range(HOURS_PER_DAY)), values.reindex([0])])
    line, fill = CLOCK_STYLES[kind]
    return go.Scatterpolar(
        r=closed.to_numpy(), theta=[hour * DEGREES_PER_HOUR for hour in closed.index],
        name=kind, mode="lines", line={"color": line, "width": 3, "shape": "spline"},
        fill="toself", fillcolor=fill,
        customdata=closed.index, hovertemplate="%{customdata}:00 — %{r:.2f} kWh<extra>" + kind + "</extra>",
    )


def clock_polar(radial_max: float | None = None) -> dict:
    radial = {"gridcolor": GRID, "linecolor": "rgba(0,0,0,0)", "showticklabels": False, "nticks": 4}
    if radial_max is not None:
        radial["range"] = [0, radial_max]
    return {
        "bgcolor": "rgba(0,0,0,0)",
        "angularaxis": {
            "rotation": 90, "direction": "clockwise", "gridcolor": GRID, "linecolor": GRID,
            "tickmode": "array", "tickvals": [h * DEGREES_PER_HOUR for h in range(0, 24, 3)],
            "ticktext": [f"{h:02d}:00" for h in range(0, 24, 3)], "tickfont": {"color": MUTED},
        },
        "radialaxis": radial,
    }


def load_clock(hourly_f: pd.DataFrame) -> go.Figure:
    """24-hour polar profile: average kWh per hour, weekday vs weekend."""
    profile = hourly_f.groupby([day_type_of(hourly_f), "hour"])["energy_kwh"].mean()
    figure = go.Figure()
    for kind in DAY_TYPES:
        if kind in profile.index.get_level_values(0):
            figure.add_trace(clock_trace(profile.loc[kind], kind))
    figure.update_layout(
        height=460, showlegend=True, polar=clock_polar(),
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.06},
        margin={"l": 30, "r": 30, "t": 40, "b": 70},
        annotations=[{"text": CLOCK_NOTE, "showarrow": False, "x": 0.5, "y": -0.2,
                      "xref": "paper", "yref": "paper", "font": {"size": 11, "color": MUTED}}],
    )
    return figure


def year_on_the_clock(hourly_f: pd.DataFrame) -> go.Figure:
    """The load clock animated month by month; the radius is fixed so seasons swell and shrink."""
    profile = hourly_f.groupby(["month", day_type_of(hourly_f), "hour"])["energy_kwh"].mean()
    months = sorted(profile.index.get_level_values(0).unique())
    kinds = [kind for kind in DAY_TYPES if kind in profile.index.get_level_values(1)]

    def traces(month: int) -> list[go.Scatterpolar]:
        per_month = profile.loc[month]
        return [clock_trace(per_month.loc[kind] if kind in per_month.index.get_level_values(0)
                            else pd.Series(dtype="float64"), kind) for kind in kinds]

    frames = [go.Frame(data=traces(month), name=MONTH_NAMES[month - 1],
                       layout={"title": {"text": MONTH_NAMES[month - 1]}}) for month in months]
    figure = go.Figure(data=traces(months[0]), frames=frames)
    step_args = {"frame": {"duration": 650, "redraw": True}, "transition": {"duration": 350},
                 "mode": "immediate"}
    figure.update_layout(
        height=560, polar=clock_polar(float(profile.max()) * 1.08),
        title={"text": MONTH_NAMES[months[0] - 1], "x": 0.5, "xanchor": "center",
               "font": {"family": FONT_DISPLAY, "size": 22, "color": GLOW}},
        legend={"orientation": "v", "x": 1.0, "xanchor": "right", "y": 1.12},
        margin={"l": 30, "r": 30, "t": 70, "b": 90},
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.0, "y": 1.12, "xanchor": "left",
            "bgcolor": PANEL, "bordercolor": AMBER, "font": {"color": TEXT},
            "buttons": [
                {"label": "▶  Play the year", "method": "animate", "args": [None, {**step_args, "fromcurrent": True}]},
                {"label": "❚❚  Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.05, "len": 0.9, "y": -0.08, "pad": {"t": 10},
            "currentvalue": {"visible": False}, "bgcolor": PANEL, "activebgcolor": AMBER,
            "bordercolor": GRID, "tickcolor": MUTED, "font": {"color": MUTED, "size": 11},
            "steps": [{"label": MONTH_NAMES[month - 1][:3], "method": "animate",
                       "args": [[MONTH_NAMES[month - 1]], step_args]} for month in months],
        }],
    )
    return figure


def calendar_heatmap(daily_f: pd.DataFrame) -> go.Figure:
    """Every selected day as a cell: week of the year across, weekday down, one row per year."""
    days = daily_f["energy_kwh"]
    years = sorted(days.index.year.unique())
    figure = make_subplots(rows=len(years), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                           subplot_titles=[str(year) for year in years])
    for row, year in enumerate(years, start=1):
        part = days[days.index.year == year]
        jan_first = pd.Timestamp(year=year, month=1, day=1)
        week = ((part.index - jan_first).days + jan_first.day_of_week) // config.WEEK_DAYS
        figure.add_trace(go.Heatmap(
            x=week, y=part.index.day_of_week, z=part.to_numpy(), customdata=part.index.strftime("%a %d %b %Y"),
            coloraxis="coloraxis", xgap=2, ygap=2,
            hovertemplate="%{customdata}: %{z:.1f} kWh<extra></extra>",
        ), row=row, col=1)
        figure.update_yaxes(row=row, col=1, autorange="reversed", tickmode="array",
                            tickvals=[0, 2, 4, 6], ticktext=["Mon", "Wed", "Fri", "Sun"], showgrid=False)
    figure.update_xaxes(showgrid=False, tickmode="array",
                        tickvals=[(pd.Timestamp(2009, m, 1).day_of_year + 3) // 7 for m in range(1, 13)],
                        ticktext=[name[:3] for name in MONTH_NAMES])
    august_2008 = days["2008-08-01":"2008-08-31"]
    if 2008 in years and not august_2008.empty:
        figure.add_annotation(
            text=f"August 2008: {august_2008.mean():.1f} kWh a day, the house is almost empty",
            x=33, y=-0.9, row=years.index(2008) + 1, col=1, showarrow=False, yanchor="bottom",
            font={"color": GLOW, "size": 12},
        )
    figure.update_layout(
        height=150 * len(years) + 80, title="Every day, one cell",
        coloraxis={"colorscale": HEAT_SCALE, "colorbar": {"thickness": 12, "outlinewidth": 0, "title": "kWh"}},
        margin={"l": 56, "r": 16, "t": 80, "b": 30},
    )
    for year in years:
        figure.update_annotations(selector={"text": str(year)}, font={"color": MUTED, "family": FONT_DISPLAY, "size": 13}, x=0, xanchor="left")
    return figure


def kpi(column, value: str, unit: str, label: str) -> None:
    column.markdown(
        f'<div class="kpi"><div class="kpi-value">{value}<small>{unit}</small></div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


hourly, daily = load_data()

st.sidebar.header("Filters")
years = sorted(daily["year"].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)
selected_seasons = st.sidebar.multiselect("Season", SEASON_ORDER, default=SEASON_ORDER)
selected_day_types = st.sidebar.multiselect("Day type", DAY_TYPES, default=DAY_TYPES)
meter_names = {code: name for code, (name, _) in config.SUB_METERS.items()}
selected_meters = st.sidebar.multiselect("Category (sub-meter)", list(meter_names),
                                         default=list(meter_names), format_func=meter_names.get)
st.sidebar.caption("Every number and chart on the page follows these filters.")


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    day_type = frame["is_weekend"].map({False: "weekday", True: "weekend"})
    mask = (frame["year"].isin(selected_years)
            & frame["season"].isin(selected_seasons)
            & day_type.isin(selected_day_types))
    return frame[mask]


hourly_f = apply_filters(hourly)
daily_f = apply_filters(daily)

if daily_f.empty:
    st.markdown('<div class="hero-title">No days match these filters</div>', unsafe_allow_html=True)
    st.warning("Add at least one year, season and day type in the sidebar.")
    st.stop()

peak_time = hourly_f["energy_kwh"].idxmax()
busiest_hour = int(hourly_f.groupby("hour")["energy_kwh"].mean().idxmax())

hero_text, hero_clock = st.columns([1.05, 1], gap="large")
with hero_text:
    st.markdown(
        '<div class="hero-title">One house in Sceaux, minute by minute'
        "<span>Electricity use from December 2006 to November 2010</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="hero-lede">In the <b>{len(daily_f):,} days</b> selected, the house used '
        f'<b>{daily_f["energy_kwh"].mean():.1f} kWh a day</b> on average. Its busiest hour of the day '
        f"is <b>{busiest_hour}:00</b>. The clock on the right shows the average load for every hour: "
        "weekdays rise twice, in the morning and in the evening; weekends stay high all afternoon.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    kpi_columns = st.columns(2)
    kpi(kpi_columns[0], f"{len(daily_f):,}", "days", "Days selected")
    kpi(kpi_columns[1], f"{daily_f['energy_kwh'].mean():.1f}", "kWh", "Average per day")
    kpi_columns = st.columns(2)
    kpi(kpi_columns[0], f"{daily_f['energy_kwh'].max():.1f}", "kWh",
        f"Max day, {daily_f['energy_kwh'].idxmax():%d %b %Y}")
    kpi(kpi_columns[1], f"{hourly_f['energy_kwh'].max():.2f}", "kWh",
        f"Peak hour, {peak_time:%d %b %Y %H}:00")
with hero_clock:
    show(load_clock(hourly_f))

st.write("")
tab_trend, tab_profiles, tab_meters, tab_forecast = st.tabs(
    ["Trend", "Load profiles", "Sub-meters", "Forecast"])

with tab_trend:
    window = st.slider("Moving average window, days", 1, 60, 7)
    trend = daily_f[["energy_kwh"]].copy()
    trend["moving_average"] = trend["energy_kwh"].rolling(window, min_periods=1).mean()
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=trend.index, y=trend["energy_kwh"], name="Daily kWh", mode="lines",
        line={"color": "rgba(242,163,58,0.55)", "width": 1}, fill="tozeroy",
        fillcolor="rgba(242,163,58,0.12)", hovertemplate="%{x|%d %b %Y}: %{y:.1f} kWh<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=trend.index, y=trend["moving_average"], name=f"{window}-day average", mode="lines",
        line={"color": GLOW, "width": 2.6}, hovertemplate="%{y:.1f} kWh<extra>average</extra>",
    ))
    figure.update_layout(title="Daily consumption", height=440, hovermode="x unified",
                         yaxis_title="kWh per day", legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
    show(figure)
    show(calendar_heatmap(daily_f))

with tab_profiles:
    st.markdown('<p class="section-note">Press play: the clock swells in winter and shrinks in summer, '
                "while the evening peak keeps its place.</p>", unsafe_allow_html=True)
    show(year_on_the_clock(hourly_f))
    left, right = st.columns(2)
    profile = hourly_f.groupby(["hour", "season"])["energy_kwh"].mean().reset_index()
    figure = px.line(profile, x="hour", y="energy_kwh", color="season", line_shape="spline",
                     category_orders={"season": SEASON_ORDER}, color_discrete_map=SEASON_COLORS,
                     labels={"energy_kwh": "Average kWh per hour", "hour": "Hour of day"},
                     title="Daily load profile by season")
    figure.update_traces(line={"width": 3})
    figure.update_layout(height=400, legend_title_text="")
    with left:
        show(figure)
    weekday = daily_f.groupby("weekday")["energy_kwh"].mean().reset_index()
    weekday["weekday"] = weekday["weekday"].map(dict(enumerate(timeseries.WEEKDAY_NAMES)))
    weekday["kind"] = weekday["weekday"].isin(["Sat", "Sun"]).map({True: "weekend", False: "weekday"})
    figure = px.bar(weekday, x="weekday", y="energy_kwh", color="kind",
                    color_discrete_map={"weekday": STEEL, "weekend": AMBER},
                    labels={"energy_kwh": "Average kWh per day", "weekday": ""}, title="Weekly profile")
    figure.update_traces(marker_line_width=0)
    figure.update_layout(height=400, bargap=0.35, showlegend=False)
    with right:
        show(figure)
    heat = hourly_f.groupby(["hour", "month"])["energy_kwh"].mean().unstack()
    figure = px.imshow(heat, aspect="auto", color_continuous_scale=HEAT_SCALE,
                       labels={"x": "Month", "y": "Hour", "color": "kWh"},
                       title="Average load by hour and month")
    figure.update_layout(height=470, coloraxis_colorbar={"thickness": 12, "outlinewidth": 0})
    show(figure)

with tab_meters:
    if not selected_meters:
        st.info("Select at least one sub-meter in the sidebar.")
    else:
        monthly = daily_f[selected_meters].resample("ME").sum() / config.WATTS_PER_KILOWATT
        monthly = monthly.rename(columns=meter_names)
        left, right = st.columns([1.7, 1])
        figure = px.bar(monthly, color_discrete_map=METER_COLORS,
                        labels={"value": "kWh", "datetime": "Month", "variable": "Sub-meter"},
                        title="Monthly energy by sub-meter")
        figure.update_traces(marker_line_width=0)
        figure.update_layout(height=430, bargap=0.15, legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
        with left:
            show(figure)
        shares = monthly.sum()
        figure = go.Figure(go.Pie(
            values=shares.values, labels=shares.index, hole=0.62, sort=False,
            marker={"colors": [METER_COLORS[name] for name in shares.index], "line": {"color": INK, "width": 3}},
            textinfo="percent", hovertemplate="%{label}: %{value:,.0f} kWh<extra></extra>",
        ))
        figure.update_layout(title="Share of energy", height=430, showlegend=True,
                             legend={"orientation": "h", "y": -0.05},
                             annotations=[{"text": f"{shares.sum():,.0f}<br>kWh", "showarrow": False,
                                           "font": {"family": FONT_DISPLAY, "size": 20, "color": TEXT}}])
        with right:
            show(figure)

with tab_forecast:
    metrics, predictions = load_forecasts()
    st.subheader("Model comparison on 2010 hold-out (day-ahead)")
    st.markdown('<p class="section-note">Trained on 2006–2009, tested on 329 days of 2010. '
                "Lower MAE is better; MAE vs naive below 1 means the model beats the baseline.</p>",
                unsafe_allow_html=True)
    st.dataframe(
        metrics, hide_index=True, width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "MAE": st.column_config.ProgressColumn("MAE, kWh/day", format="%.2f",
                                                   min_value=0.0, max_value=float(metrics["MAE"].max())),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "MAPE_pct": st.column_config.NumberColumn("MAPE, %", format="%.1f"),
            "MASE_vs_naive": st.column_config.NumberColumn("MAE vs naive", format="%.2f"),
            "fit_time_s": st.column_config.NumberColumn("Time, s", format="%.4f"),
            "peak_memory_mb": st.column_config.NumberColumn("Memory, MB", format="%.2f"),
        },
    )
    models = st.multiselect("Models to plot", list(predictions.columns.drop("actual")),
                            default=["Naive (baseline)", "Linear regression (lags)"])
    figure = go.Figure()
    for name in ["actual"] + models:
        figure.add_trace(go.Scatter(
            x=predictions.index, y=predictions[name], name=name, mode="lines",
            line={"color": MODEL_COLORS.get(name, STEEL), "width": 1.4 if name == "actual" else 2.2,
                  "dash": "dot" if name == "Naive (baseline)" else "solid"},
            opacity=0.9 if name == "actual" else 1.0,
        ))
    figure.update_layout(title="Forecast vs actual", height=440, hovermode="x unified",
                         yaxis_title="kWh per day", legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
    show(figure)
    horizon = st.number_input("Forecast next N days", min_value=1, max_value=30, value=7)
    future = forecast.forecast_future(daily["energy_kwh"].astype("float64"), int(horizon))
    history = daily["energy_kwh"].iloc[-30:]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history.index, y=history.values, name="Last 30 days", mode="lines+markers",
                                line={"color": STEEL, "width": 2}, marker={"size": 5}))
    figure.add_trace(go.Bar(x=future.index, y=future.values, name="Forecast",
                            marker={"color": AMBER, "line": {"width": 0}},
                            hovertemplate="%{x|%d %b %Y}: %{y:.1f} kWh<extra>forecast</extra>"))
    figure.update_layout(title="Next days (linear regression, recursive)", height=400, bargap=0.25,
                         yaxis_title="kWh per day", legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
    show(figure)
