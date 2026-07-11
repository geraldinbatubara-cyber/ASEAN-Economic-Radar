from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


ASEAN_COUNTRIES = {
    "BRN": "Brunei Darussalam",
    "KHM": "Cambodia",
    "IDN": "Indonesia",
    "LAO": "Lao PDR",
    "MYS": "Malaysia",
    "MMR": "Myanmar",
    "PHL": "Philippines",
    "SGP": "Singapore",
    "THA": "Thailand",
    "VNM": "Vietnam",
}


@dataclass(frozen=True)
class Indicator:
    code: str
    label: str
    unit: str
    pillar: str
    higher_is_better: bool
    short_note: str


INDICATORS = [
    Indicator("NY.GDP.MKTP.CD", "GDP current US$", "US$", "Market Scale", True, "Nominal size of the economy."),
    Indicator("NY.GDP.PCAP.CD", "GDP per capita current US$", "US$ per person", "Income", True, "Average income proxy."),
    Indicator("NY.GDP.MKTP.KD.ZG", "GDP growth", "%", "Growth Momentum", True, "Annual real GDP growth."),
    Indicator("FP.CPI.TOTL.ZG", "Inflation", "%", "Stability", False, "Consumer price inflation."),
    Indicator("SL.UEM.TOTL.ZS", "Unemployment", "% of labor force", "Labour", False, "Total unemployment rate."),
    Indicator("SP.POP.TOTL", "Population", "people", "Market Scale", True, "Domestic market size."),
    Indicator("NE.TRD.GNFS.ZS", "Trade openness", "% of GDP", "Openness", True, "Exports plus imports as share of GDP."),
    Indicator("BX.KLT.DINV.CD.WD", "FDI net inflows", "US$", "Investment", True, "Foreign direct investment inflows."),
    Indicator("NE.EXP.GNFS.CD", "Exports of goods and services", "US$", "External Sector", True, "Export value."),
    Indicator("NE.IMP.GNFS.CD", "Imports of goods and services", "US$", "External Sector", True, "Import value."),
    Indicator("BN.CAB.XOKA.GD.ZS", "Current account balance", "% of GDP", "External Balance", True, "Current account balance relative to GDP."),
    Indicator("NE.GDI.TOTL.ZS", "Gross capital formation", "% of GDP", "Investment", True, "Domestic investment relative to GDP."),
    Indicator("NY.GNS.ICTR.ZS", "Gross savings", "% of GDP", "Savings", True, "Gross savings relative to GDP."),
    Indicator("NV.IND.TOTL.ZS", "Industry value added", "% of GDP", "Economic Structure", True, "Industrial sector share of GDP."),
    Indicator("NV.SRV.TOTL.ZS", "Services value added", "% of GDP", "Economic Structure", True, "Services sector share of GDP."),
    Indicator("NV.AGR.TOTL.ZS", "Agriculture value added", "% of GDP", "Economic Structure", False, "Agriculture sector share of GDP."),
    Indicator("GC.TAX.TOTL.GD.ZS", "Tax revenue", "% of GDP", "Fiscal Capacity", True, "Tax revenue relative to GDP."),
    Indicator("GC.DOD.TOTL.GD.ZS", "Central government debt", "% of GDP", "Fiscal Risk", False, "Central government debt relative to GDP."),
    Indicator("IT.NET.USER.ZS", "Individuals using the Internet", "% of population", "Digital Readiness", True, "Internet usage as a digital readiness proxy."),
]

INDICATOR_BY_LABEL = {item.label: item for item in INDICATORS}
COUNTRY_PARAM = ";".join(ASEAN_COUNTRIES)
WB_API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"


st.set_page_config(
    page_title="ASEAN Economic Radar",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(31, 41, 51, 0.08);
            padding: 0.9rem 1rem;
            border-radius: 8px;
        }
        div[data-testid="stMetricLabel"] {color: #52606D;}
        .source-note {
            color: #52606D;
            font-size: 0.92rem;
            border-left: 3px solid #0E7C7B;
            padding-left: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_number(value: float | int | None, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if unit == "US$":
        if abs(value) >= 1_000_000_000_000:
            return f"US$ {value / 1_000_000_000_000:,.2f} T"
        if abs(value) >= 1_000_000_000:
            return f"US$ {value / 1_000_000_000:,.1f} B"
        if abs(value) >= 1_000_000:
            return f"US$ {value / 1_000_000:,.1f} M"
        return f"US$ {value:,.0f}"
    if unit == "US$ per person":
        return f"US$ {value:,.0f}"
    if unit == "people":
        if value >= 1_000_000:
            return f"{value / 1_000_000:,.1f} M"
        return f"{value:,.0f}"
    if unit.startswith("%") or unit == "%":
        return f"{value:,.2f}%"
    return f"{value:,.2f}"


def fetch_world_bank_indicator(indicator: Indicator, start_year: int, end_year: int) -> pd.DataFrame:
    params = {
        "format": "json",
        "per_page": 20000,
        "date": f"{start_year}:{end_year}",
    }
    url = WB_API.format(countries=COUNTRY_PARAM, indicator=indicator.code)
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []

    records = []
    for row in rows:
        country = row.get("country", {})
        iso3 = row.get("countryiso3code")
        if iso3 not in ASEAN_COUNTRIES:
            continue
        records.append(
            {
                "iso3": iso3,
                "country": ASEAN_COUNTRIES[iso3],
                "year": int(row["date"]),
                "indicator_code": indicator.code,
                "indicator": indicator.label,
                "unit": indicator.unit,
                "pillar": indicator.pillar,
                "higher_is_better": indicator.higher_is_better,
                "source_country_label": country.get("value", ASEAN_COUNTRIES[iso3]),
                "value": row.get("value"),
            }
        )
    return pd.DataFrame.from_records(records)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_wdi_data(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    failures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(fetch_world_bank_indicator, indicator, start_year, end_year): indicator
            for indicator in INDICATORS
        }
        for future in as_completed(future_map):
            indicator = future_map[future]
            try:
                frames.append(future.result())
            except Exception as exc:  # pragma: no cover - visible in Streamlit app
                failures.append(f"{indicator.label}: {exc}")

    if not frames:
        raise RuntimeError(
            "World Bank API could not return data for the selected indicators. "
            "Try refreshing the app or selecting a shorter year range."
        )

    data = pd.concat(frames, ignore_index=True)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"]).sort_values(["country", "indicator", "year"])
    data.attrs["failures"] = failures
    return data


def latest_observations(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    latest_idx = data.sort_values("year").groupby(["iso3", "indicator"], as_index=False).tail(1).index
    latest = data.loc[latest_idx].copy()
    latest["formatted_value"] = latest.apply(lambda row: fmt_number(row["value"], row["unit"]), axis=1)
    return latest.sort_values(["indicator", "value"], ascending=[True, False])


def normalize_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series([50.0] * len(values), index=values.index)
    score = (values - min_value) / (max_value - min_value) * 100
    if not higher_is_better:
        score = 100 - score
    return score


def build_radar_index(latest: pd.DataFrame) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()

    parts = []
    for indicator_label, group in latest.groupby("indicator"):
        indicator = INDICATOR_BY_LABEL[indicator_label]
        part = group[["iso3", "country", "indicator", "pillar", "value", "unit", "year"]].copy()
        part["indicator_score"] = normalize_series(part["value"], indicator.higher_is_better)
        parts.append(part)

    scores = pd.concat(parts, ignore_index=True)
    pillar_scores = (
        scores.groupby(["iso3", "country", "pillar"], as_index=False)["indicator_score"]
        .mean()
        .rename(columns={"indicator_score": "pillar_score"})
    )
    overall = (
        pillar_scores.groupby(["iso3", "country"], as_index=False)["pillar_score"]
        .mean()
        .rename(columns={"pillar_score": "radar_score"})
        .sort_values("radar_score", ascending=False)
    )
    overall["rank"] = range(1, len(overall) + 1)
    return overall.merge(pillar_scores, on=["iso3", "country"], how="left")


def latest_metric(latest: pd.DataFrame, label: str, country: str | None = None) -> pd.Series | None:
    rows = latest[latest["indicator"].eq(label)]
    if country:
        rows = rows[rows["country"].eq(country)]
    if rows.empty:
        return None
    if label in {"Inflation", "Unemployment"}:
        return rows.loc[rows["value"].idxmin()]
    return rows.loc[rows["value"].idxmax()]


def selected_data(data: pd.DataFrame, countries: Iterable[str], indicator_labels: Iterable[str]) -> pd.DataFrame:
    return data[data["country"].isin(countries) & data["indicator"].isin(indicator_labels)].copy()


def build_radar_figure(country_scores: pd.DataFrame, country: str) -> go.Figure:
    ordered = country_scores.sort_values("pillar")
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=ordered["pillar_score"],
            theta=ordered["pillar"],
            fill="toself",
            name=country,
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        title=f"Profil pilar ekonomi: {country}",
    )
    return fig


def country_radar_summary(radar: pd.DataFrame, country: str) -> pd.Series | None:
    rows = radar[radar["country"].eq(country)].drop_duplicates("country")
    if rows.empty:
        return None
    return rows.iloc[0]


def clean_series(data: pd.DataFrame, country: str, indicator_label: str) -> pd.DataFrame:
    series = data[data["country"].eq(country) & data["indicator"].eq(indicator_label)].copy()
    series = series[["year", "value"]].dropna().sort_values("year")
    return series.drop_duplicates("year", keep="last")


def fit_ols_forecast(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) < 3:
        return pd.DataFrame(), pd.DataFrame()
    years = series["year"].to_numpy(dtype=float)
    values = series["value"].to_numpy(dtype=float)
    x_center = years - years.mean()
    design = np.column_stack([np.ones(len(x_center)), x_center])
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coef
    future_years = np.arange(int(years.max()) + 1, int(years.max()) + horizon + 1)
    future_design = np.column_stack([np.ones(len(future_years)), future_years - years.mean()])
    forecast = future_design @ coef
    fitted_df = pd.DataFrame({"year": years.astype(int), "value": fitted, "series": "OLS trend fitted"})
    forecast_df = pd.DataFrame({"year": future_years.astype(int), "value": forecast, "series": "OLS trend forecast"})
    return fitted_df, forecast_df


def make_lag_features(values: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    targets = []
    target_years = []
    for idx in range(3, len(values)):
        prev = values[idx - 1]
        lag2 = values[idx - 2]
        lag3 = values[idx - 3]
        rolling_mean = np.mean(values[idx - 3 : idx])
        momentum = prev - lag2
        rows.append([years[idx], prev, rolling_mean, momentum, lag2 - lag3])
        targets.append(values[idx])
        target_years.append(years[idx])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), np.asarray(target_years, dtype=int)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1
    X_scaled = (X - mean) / scale
    design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, mean, scale


def predict_ridge(X: np.ndarray, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    X_scaled = (X - mean) / scale
    design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    return design @ coef


def fit_ml_forecast(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) < 7:
        return pd.DataFrame(), pd.DataFrame()
    years = series["year"].to_numpy(dtype=float)
    values = series["value"].to_numpy(dtype=float)
    X, y, target_years = make_lag_features(values, years)
    if len(y) < 4:
        return pd.DataFrame(), pd.DataFrame()
    coef, mean, scale = fit_ridge(X, y, alpha=1.2)
    fitted = predict_ridge(X, coef, mean, scale)

    forecast_rows = []
    rolling_values = values.copy()
    last_year = int(years.max())
    for step in range(1, horizon + 1):
        next_year = last_year + step
        prev = rolling_values[-1]
        lag2 = rolling_values[-2]
        lag3 = rolling_values[-3]
        next_features = np.asarray([[next_year, prev, np.mean(rolling_values[-3:]), prev - lag2, lag2 - lag3]])
        next_value = float(predict_ridge(next_features, coef, mean, scale)[0])
        forecast_rows.append({"year": next_year, "value": next_value, "series": "Ridge ML forecast"})
        rolling_values = np.append(rolling_values, next_value)

    fitted_df = pd.DataFrame({"year": target_years, "value": fitted, "series": "Ridge ML fitted"})
    forecast_df = pd.DataFrame(forecast_rows)
    return fitted_df, forecast_df


def model_mae(actual: pd.DataFrame, fitted: pd.DataFrame) -> float | None:
    if actual.empty or fitted.empty:
        return None
    merged = actual.merge(fitted, on="year", suffixes=("_actual", "_fitted"))
    if merged.empty:
        return None
    return float((merged["value_actual"] - merged["value_fitted"]).abs().mean())


def build_forecast_frame(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, float | None]]:
    historical = series.copy()
    historical["series"] = "Historical"
    ols_fitted, ols_forecast = fit_ols_forecast(series, horizon)
    ml_fitted, ml_forecast = fit_ml_forecast(series, horizon)
    frames = [historical, ols_fitted, ols_forecast, ml_fitted, ml_forecast]
    forecast_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    metrics = {
        "OLS trend MAE": model_mae(series, ols_fitted),
        "Ridge ML MAE": model_mae(series, ml_fitted),
    }
    return forecast_frame, metrics


def render_forecast_chart(series_frame: pd.DataFrame, country: str, indicator: Indicator) -> None:
    if series_frame.empty:
        st.warning("Data historis belum cukup untuk membuat prediksi.")
        return
    fig = px.line(
        series_frame,
        x="year",
        y="value",
        color="series",
        markers=True,
        labels={"year": "Year", "value": indicator.unit, "series": "Model"},
        title=f"{country} - {indicator.label}",
    )
    fig.update_layout(legend_orientation="h", legend_y=-0.25)
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_summary(metrics: dict[str, float | None], unit: str) -> None:
    cols = st.columns(2)
    ols_mae = metrics.get("OLS trend MAE")
    ml_mae = metrics.get("Ridge ML MAE")
    cols[0].metric("OLS fitted MAE", fmt_number(ols_mae, unit) if ols_mae is not None else "n/a")
    cols[1].metric("Ridge ML fitted MAE", fmt_number(ml_mae, unit) if ml_mae is not None else "n/a")


def render_header(data: pd.DataFrame, latest: pd.DataFrame) -> None:
    min_year, max_year = int(data["year"].min()), int(data["year"].max())
    st.title("ASEAN Economic Radar")
    st.caption("Monitoring dan pembandingan indikator ekonomi 10 negara ASEAN berbasis data resmi publik.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Negara ASEAN", f"{len(ASEAN_COUNTRIES)}")
    col2.metric("Indikator WDI", f"{latest['indicator'].nunique()}")
    col3.metric("Rentang data", f"{min_year}-{max_year}")
    col4.metric("Observasi", f"{len(data):,}")


def render_snapshot(data: pd.DataFrame, latest: pd.DataFrame, radar: pd.DataFrame) -> None:
    st.subheader("Regional Snapshot")
    top_growth = latest_metric(latest, "GDP growth")
    lowest_inflation = latest_metric(latest, "Inflation")
    top_fdi = latest_metric(latest, "FDI net inflows")
    top_score = radar.drop_duplicates("country").sort_values("radar_score", ascending=False).head(1)

    cols = st.columns(4)
    if top_score.empty:
        cols[0].metric("Top Radar Score", "n/a")
    else:
        row = top_score.iloc[0]
        cols[0].metric("Top Radar Score", row["country"], f"{row['radar_score']:.1f}/100")
    if top_growth is not None:
        cols[1].metric("Growth tertinggi", top_growth["country"], fmt_number(top_growth["value"], "%"))
    if lowest_inflation is not None:
        cols[2].metric("Inflasi terendah", lowest_inflation["country"], fmt_number(lowest_inflation["value"], "%"))
    if top_fdi is not None:
        cols[3].metric("FDI inflow tertinggi", top_fdi["country"], fmt_number(top_fdi["value"], "US$"))

    left, right = st.columns([1.2, 1])
    with left:
        score_table = radar.drop_duplicates("country").sort_values("radar_score", ascending=False)
        fig = px.bar(
            score_table,
            x="radar_score",
            y="country",
            orientation="h",
            color="radar_score",
            color_continuous_scale="Teal",
            labels={"radar_score": "Radar score", "country": ""},
            title="ASEAN Economic Radar Score",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        growth = latest[latest["indicator"].eq("GDP growth")].sort_values("value", ascending=False)
        fig = px.bar(
            growth,
            x="country",
            y="value",
            color="value",
            color_continuous_scale="Viridis",
            labels={"value": "GDP growth (%)", "country": ""},
            title="Pertumbuhan GDP terbaru",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p class="source-note">Radar score adalah indeks komparatif internal: setiap indikator dinormalisasi 0-100 lintas negara, lalu dirata-ratakan per pilar. Ini alat pembanding, bukan peringkat resmi ASEAN.</p>',
        unsafe_allow_html=True,
    )


def render_country_compare(data: pd.DataFrame) -> None:
    st.subheader("Country Compare")
    default_countries = ["Indonesia", "Malaysia", "Singapore", "Thailand", "Vietnam"]
    countries = st.multiselect(
        "Pilih negara",
        options=list(ASEAN_COUNTRIES.values()),
        default=default_countries,
    )
    indicator_label = st.selectbox(
        "Pilih indikator",
        options=[item.label for item in INDICATORS],
        index=2,
    )
    chart_data = selected_data(data, countries, [indicator_label])
    indicator = INDICATOR_BY_LABEL[indicator_label]

    if chart_data.empty:
        st.warning("Tidak ada data untuk pilihan ini.")
        return

    fig = px.line(
        chart_data.sort_values("year"),
        x="year",
        y="value",
        color="country",
        markers=True,
        labels={"value": indicator.unit, "year": "Year", "country": "Country"},
        title=f"Tren {indicator.label}",
    )
    st.plotly_chart(fig, use_container_width=True)

    latest = latest_observations(chart_data)
    table = latest[["country", "year", "formatted_value"]].rename(
        columns={"country": "Country", "year": "Latest Year", "formatted_value": "Latest Value"}
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_indicator_explorer(data: pd.DataFrame, latest: pd.DataFrame) -> None:
    st.subheader("Indicator Explorer")
    indicator_labels = st.multiselect(
        "Pilih indikator untuk heatmap",
        options=[item.label for item in INDICATORS],
        default=["GDP growth", "Inflation", "Trade openness", "FDI net inflows"],
    )
    heat = latest[latest["indicator"].isin(indicator_labels)].copy()
    if heat.empty:
        st.warning("Tidak ada data untuk indikator yang dipilih.")
        return

    heat["score"] = 0.0
    for indicator_label, group in heat.groupby("indicator"):
        indicator = INDICATOR_BY_LABEL[indicator_label]
        heat.loc[group.index, "score"] = normalize_series(group["value"], indicator.higher_is_better)

    fig = px.imshow(
        heat.pivot_table(index="country", columns="indicator", values="score"),
        color_continuous_scale="Teal",
        aspect="auto",
        labels={"color": "Score 0-100"},
        title="Heatmap skor relatif per indikator",
    )
    st.plotly_chart(fig, use_container_width=True)

    detail = heat[["country", "indicator", "year", "formatted_value", "pillar"]].rename(
        columns={
            "country": "Country",
            "indicator": "Indicator",
            "year": "Latest Year",
            "formatted_value": "Latest Value",
            "pillar": "Pillar",
        }
    )
    st.dataframe(detail.sort_values(["Indicator", "Country"]), use_container_width=True, hide_index=True)


def render_radar_index(radar: pd.DataFrame) -> None:
    st.subheader("Economic Radar Index")
    left_country, right_country = st.columns(2)
    country_a = left_country.selectbox("Negara A", options=list(ASEAN_COUNTRIES.values()), index=2)
    country_b = right_country.selectbox("Negara B", options=list(ASEAN_COUNTRIES.values()), index=7)

    scores_a = radar[radar["country"].eq(country_a)].copy()
    scores_b = radar[radar["country"].eq(country_b)].copy()
    if scores_a.empty or scores_b.empty:
        st.warning("Skor belum tersedia untuk salah satu negara.")
        return

    summary_a = country_radar_summary(radar, country_a)
    summary_b = country_radar_summary(radar, country_b)
    metric_a, metric_b = st.columns(2)
    if summary_a is not None:
        metric_a.metric(f"{country_a} Radar Score", f"{summary_a['radar_score']:.1f}/100", f"Rank {int(summary_a['rank'])}")
    if summary_b is not None:
        metric_b.metric(f"{country_b} Radar Score", f"{summary_b['radar_score']:.1f}/100", f"Rank {int(summary_b['rank'])}")

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.plotly_chart(build_radar_figure(scores_a, country_a), use_container_width=True)
    with chart_b:
        st.plotly_chart(build_radar_figure(scores_b, country_b), use_container_width=True)

    comparison = scores_a[["pillar", "pillar_score"]].merge(
        scores_b[["pillar", "pillar_score"]],
        on="pillar",
        suffixes=(f" - {country_a}", f" - {country_b}"),
    )
    st.dataframe(comparison.sort_values("pillar"), use_container_width=True, hide_index=True)

    leaderboard = radar.drop_duplicates("country").sort_values("radar_score", ascending=False)
    st.dataframe(
        leaderboard[["rank", "country", "radar_score"]].rename(
            columns={"rank": "Rank", "country": "Country", "radar_score": "Radar Score"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_lab(data: pd.DataFrame) -> None:
    st.subheader("Prediction Lab")
    st.caption(
        "Perbandingan proyeksi sederhana: OLS trend sebagai model ekonometrika ringan dan Ridge autoregression sebagai model ML ringan."
    )

    controls = st.columns([1, 1, 1])
    country_a = controls[0].selectbox("Negara A", options=list(ASEAN_COUNTRIES.values()), index=2, key="pred_country_a")
    country_b = controls[1].selectbox("Negara B", options=list(ASEAN_COUNTRIES.values()), index=9, key="pred_country_b")
    forecast_indicators = [
        "GDP growth",
        "Inflation",
        "Trade openness",
        "FDI net inflows",
        "GDP per capita current US$",
        "Current account balance",
        "Gross capital formation",
        "Gross savings",
        "Tax revenue",
        "Individuals using the Internet",
    ]
    available_labels = [item.label for item in INDICATORS if item.label in forecast_indicators]
    indicator_label = controls[2].selectbox("Indikator prediksi", options=available_labels, index=0)
    horizon = st.slider("Horizon prediksi", min_value=1, max_value=5, value=3)
    indicator = INDICATOR_BY_LABEL[indicator_label]

    left, right = st.columns(2)
    for column, country in [(left, country_a), (right, country_b)]:
        series = clean_series(data, country, indicator_label)
        forecast_frame, metrics = build_forecast_frame(series, horizon)
        with column:
            render_forecast_chart(forecast_frame, country, indicator)
            render_forecast_summary(metrics, indicator.unit)
            future = forecast_frame[forecast_frame["series"].str.contains("forecast", case=False, na=False)].copy()
            if not future.empty:
                future["formatted_value"] = future["value"].apply(lambda value: fmt_number(value, indicator.unit))
                st.dataframe(
                    future[["year", "series", "formatted_value"]].rename(
                        columns={"year": "Year", "series": "Model", "formatted_value": "Forecast"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown(
        '<p class="source-note">Prediction Lab adalah eksplorasi skenario berbasis tren historis WDI. OLS dan Ridge ML tidak menggantikan proyeksi resmi IMF, World Bank, atau otoritas nasional.</p>',
        unsafe_allow_html=True,
    )


def render_methodology(failures: list[str]) -> None:
    st.subheader("Data Sources & Methodology")
    st.markdown(
        """
        **Sumber live MVP:** World Bank Open Data / World Development Indicators API.

        **Cakupan negara:** seluruh anggota ASEAN kecuali Timor-Leste: Brunei Darussalam, Cambodia, Indonesia,
        Lao PDR, Malaysia, Myanmar, Philippines, Singapore, Thailand, dan Vietnam.

        **Metode indeks:** indikator terbaru yang tersedia untuk setiap negara dinormalisasi 0-100. Untuk indikator
        stabilitas seperti inflasi dan pengangguran, nilai lebih rendah diberi skor lebih tinggi. Skor pilar adalah
        rata-rata indikator dalam pilar, dan radar score adalah rata-rata seluruh pilar tersedia.

        **Prediction Lab:** proyeksi dibuat sebagai pembanding eksploratif. Model ekonometrika memakai OLS trend
        tahunan, sedangkan model ML memakai ridge autoregression berbasis lag, rolling mean, dan momentum historis.
        Output ini bukan proyeksi resmi.

        **Catatan:** tahun terbaru dapat berbeda antar negara/indikator karena jadwal rilis statistik resmi tidak seragam.
        """
    )
    st.link_button("World Bank Open Data", "https://data.worldbank.org/")
    st.link_button("ASEANstats Data Portal", "https://data.aseanstats.org/")
    st.link_button("IMF Data Portal", "https://data.imf.org/")
    st.link_button("UN Comtrade", "https://comtradeplus.un.org/")

    if failures:
        with st.expander("Catatan pengambilan data"):
            for failure in failures:
                st.write(f"- {failure}")


def main() -> None:
    inject_style()
    st.sidebar.title("ASEAN Economic Radar")
    start_year, end_year = st.sidebar.slider("Rentang tahun", 2010, 2026, (2015, 2025))
    page = st.sidebar.radio(
        "Navigasi",
        [
            "ASEAN Snapshot",
            "Country Compare",
            "Indicator Explorer",
            "Economic Radar Index",
            "Prediction Lab",
            "Sources & Methodology",
        ],
    )

    with st.spinner("Mengambil data resmi publik dari World Bank Open Data..."):
        data = load_wdi_data(start_year, end_year)

    failures = data.attrs.get("failures", [])
    latest = latest_observations(data)
    radar = build_radar_index(latest)
    render_header(data, latest)

    if page == "ASEAN Snapshot":
        render_snapshot(data, latest, radar)
    elif page == "Country Compare":
        render_country_compare(data)
    elif page == "Indicator Explorer":
        render_indicator_explorer(data, latest)
    elif page == "Economic Radar Index":
        render_radar_index(radar)
    elif page == "Prediction Lab":
        render_prediction_lab(data)
    else:
        render_methodology(failures)


if __name__ == "__main__":
    main()
