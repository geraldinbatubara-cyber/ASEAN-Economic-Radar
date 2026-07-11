from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    response = requests.get(url, params=params, timeout=30)
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
    for indicator in INDICATORS:
        try:
            frames.append(fetch_world_bank_indicator(indicator, start_year, end_year))
        except Exception as exc:  # pragma: no cover - visible in Streamlit app
            failures.append(f"{indicator.label}: {exc}")

    if not frames:
        raise RuntimeError("World Bank API could not return data for the selected indicators.")

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
    country = st.selectbox("Pilih negara untuk radar chart", options=list(ASEAN_COUNTRIES.values()), index=2)
    country_scores = radar[radar["country"].eq(country)].copy()
    if country_scores.empty:
        st.warning("Skor belum tersedia untuk negara ini.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=country_scores["pillar_score"],
            theta=country_scores["pillar"],
            fill="toself",
            name=country,
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        title=f"Profil pilar ekonomi: {country}",
    )
    st.plotly_chart(fig, use_container_width=True)

    leaderboard = radar.drop_duplicates("country").sort_values("radar_score", ascending=False)
    st.dataframe(
        leaderboard[["rank", "country", "radar_score"]].rename(
            columns={"rank": "Rank", "country": "Country", "radar_score": "Radar Score"}
        ),
        use_container_width=True,
        hide_index=True,
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
        ["ASEAN Snapshot", "Country Compare", "Indicator Explorer", "Economic Radar Index", "Sources & Methodology"],
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
    else:
        render_methodology(failures)


if __name__ == "__main__":
    main()
