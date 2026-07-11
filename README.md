# ASEAN Economic Radar

Dashboard Streamlit untuk memantau dan membandingkan indikator ekonomi 10 negara ASEAN, berbasis data resmi publik.

## Negara

MVP ini mencakup seluruh negara anggota ASEAN kecuali Timor-Leste:

- Brunei Darussalam
- Cambodia
- Indonesia
- Lao PDR
- Malaysia
- Myanmar
- Philippines
- Singapore
- Thailand
- Vietnam

## Fitur MVP

- ASEAN Snapshot: ringkasan regional, ranking pertumbuhan, dan skor komparatif.
- Country Compare: perbandingan tren antarnegara untuk setiap indikator.
- Indicator Explorer: heatmap skor relatif antarnegara.
- Economic Radar Index: indeks komparatif transparan berbasis normalisasi 0-100.
- Sources & Methodology: catatan sumber data dan metodologi.

## Sumber Data

Versi awal menggunakan World Bank Open Data / World Development Indicators API secara live saat aplikasi dijalankan.
Ekspansi berikutnya disiapkan untuk ASEANstats, IMF Data, dan UN Comtrade.

Lihat `data_sources.md` untuk detail.

## Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
