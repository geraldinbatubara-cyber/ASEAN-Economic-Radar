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
- Transport Cost & Logistics: perbandingan biaya kepatuhan ekspor/impor, LPI, dan konektivitas transportasi.
- Economic Radar Index: dua radar chart berdampingan untuk membandingkan dua negara.
- Prediction Lab: proyeksi eksploratif dengan model OLS trend dan Ridge ML autoregression.
- Sources & Methodology: catatan sumber data dan metodologi.

## Sumber Data

Versi awal menggunakan World Bank Open Data / World Development Indicators API secara live saat aplikasi dijalankan.
Ekspansi berikutnya disiapkan untuk ASEANstats, IMF Data, dan UN Comtrade.

Indikator mencakup pertumbuhan, inflasi, pengangguran, perdagangan, FDI, current account, investasi domestik,
tabungan, struktur ekonomi sektoral, fiskal, utang pemerintah pusat, kesiapan digital, biaya kepatuhan ekspor/impor,
performa logistik, dan konektivitas transportasi perdagangan.

Lihat `data_sources.md` untuk detail.

## Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
