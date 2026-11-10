"""Static (Matplotlib/Seaborn) and interactive (Plotly) charts. Every chart is saved to results/figures."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src import config

FIGURE_SIZE = (12, 5)
DPI = 120
sns.set_theme(style="whitegrid")


def _save(figure: plt.Figure, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")


def plot_boxplots(frame: pd.DataFrame, sample_size: int = 200_000) -> plt.Figure:
    """Boxplots of all numeric features (random sample for speed) to show IQR outliers."""
    sample = frame[config.NUMERIC_COLUMNS].dropna().sample(sample_size, random_state=42)
    figure, axes = plt.subplots(1, len(config.NUMERIC_COLUMNS), figsize=(16, 4))
    for axis, column in zip(axes, config.NUMERIC_COLUMNS):
        sns.boxplot(y=sample[column], ax=axis, color="#4C72B0", fliersize=1)
        axis.set_title(column.replace("_", " "), fontsize=9)
        axis.set_ylabel("")
    figure.suptitle("Distribution and IQR outliers of minute-level features")
    figure.tight_layout()
    _save(figure, "01_boxplots")
    return figure


def plot_rolling(rolling: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(rolling.index, rolling["daily_kwh"], color="lightgray", linewidth=0.7, label="Daily total")
    for column, color in zip(rolling.columns[1:], ["#4C72B0", "#DD8452", "#C44E52"]):
        axis.plot(rolling.index, rolling[column], color=color, linewidth=1.6,
                  label=column.replace("rolling_", "Moving avg ").replace("d", " days"))
    axis.set(title="Daily energy consumption and moving averages", xlabel="Date", ylabel="kWh per day")
    axis.legend()
    _save(figure, "02_rolling_means")
    return figure


def plot_profiles(daily_profile: pd.DataFrame, weekly_profile: pd.Series,
                  seasonal: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    daily_profile.plot(ax=axes[0], marker="o")
    axes[0].set(title="Daily load profile", xlabel="Hour of day", ylabel="Average kWh per hour")
    weekly_profile.plot.bar(ax=axes[1], color="#4C72B0")
    axes[1].set(title="Weekly profile", xlabel="Weekday", ylabel="Average kWh per day")
    order = ["winter", "spring", "summer", "autumn"]
    seasonal.loc[order, "mean"].plot.bar(ax=axes[2], color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"])
    axes[2].set(title="Seasonal profile", xlabel="Season", ylabel="Average kWh per day")
    for axis in axes[1:]:
        axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    _save(figure, "03_load_profiles")
    return figure


def plot_hour_month_heatmap(heatmap: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(heatmap, cmap="YlOrRd", ax=axis, cbar_kws={"label": "Average kWh per hour"})
    axis.set(title="Average load by hour and month", xlabel="Month", ylabel="Hour of day")
    _save(figure, "04_hour_month_heatmap")
    return figure


def plot_correlation(corr: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Correlation matrix (hourly data)")
    _save(figure, "05_correlation_heatmap")
    return figure


def plot_decomposition(result, title: str, name: str) -> plt.Figure:
    figure = result.plot()
    figure.set_size_inches(12, 8)
    figure.suptitle(title, y=1.01)
    _save(figure, name)
    return figure


def plot_peaks(peaks: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    by_hour = peaks["by_hour"].reindex(range(config.HOURS_PER_DAY), fill_value=0)
    by_hour.plot.bar(ax=axes[0], color="#C44E52")
    axes[0].set(title=f"Peak hours (load ≥ {peaks['threshold_kwh']} kWh, top 5%)",
                xlabel="Hour of day", ylabel="Number of peak hours")
    peaks["by_season"].plot.bar(ax=axes[1], color="#DD8452")
    axes[1].set(title="Peak hours by season", xlabel="Season", ylabel="Number of peak hours")
    axes[1].tick_params(axis="x", rotation=0)
    figure.tight_layout()
    _save(figure, "07_peak_hours")
    return figure


def plot_forecasts(predictions: pd.DataFrame, last_days: int = 90) -> plt.Figure:
    tail = predictions.tail(last_days)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(tail.index, tail["actual"], color="black", linewidth=2, label="Actual")
    for column in tail.columns.drop("actual"):
        axis.plot(tail.index, tail[column], linewidth=1.2, alpha=0.85, label=column)
    axis.set(title=f"Day-ahead forecasts vs actual (last {last_days} days of test)",
             xlabel="Date", ylabel="kWh per day")
    axis.legend(fontsize=8)
    _save(figure, "08_forecasts")
    return figure


def plot_residuals(predictions: pd.DataFrame, model: str) -> plt.Figure:
    residuals = predictions["actual"] - predictions[model]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(residuals.index, residuals, color="#4C72B0", linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title=f"Residuals over time — {model}", xlabel="Date", ylabel="Actual − predicted, kWh")
    sns.histplot(residuals, kde=True, ax=axes[1], color="#4C72B0")
    axes[1].set(title="Residual distribution", xlabel="kWh")
    figure.tight_layout()
    _save(figure, "09_residuals")
    return figure


def interactive_daily(daily: pd.DataFrame) -> go.Figure:
    """Interactive Plotly line with a range slider; saved as standalone HTML."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=daily.index, y=daily["energy_kwh"], name="Daily kWh",
                                line={"width": 1}))
    figure.add_trace(go.Scatter(x=daily.index, y=daily["energy_kwh"].rolling(30).mean(),
                                name="30-day moving average", line={"width": 2.5}))
    figure.update_layout(title="Household daily energy consumption (interactive)",
                         xaxis_title="Date", yaxis_title="kWh per day",
                         xaxis={"rangeslider": {"visible": True}}, height=500)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.write_html(config.FIGURES_DIR / "interactive_daily.html", include_plotlyjs="cdn")
    return figure


def interactive_sub_meters(daily: pd.DataFrame) -> go.Figure:
    monthly = daily[list(config.SUB_METERS)].resample("ME").sum() / config.WATTS_PER_KILOWATT
    figure = go.Figure()
    for column, (name, _) in config.SUB_METERS.items():
        figure.add_trace(go.Bar(x=monthly.index, y=monthly[column], name=name))
    figure.update_layout(barmode="stack", title="Monthly energy by sub-meter (interactive)",
                         xaxis_title="Month", yaxis_title="kWh", height=500)
    figure.write_html(config.FIGURES_DIR / "interactive_sub_meters.html", include_plotlyjs="cdn")
    return figure
