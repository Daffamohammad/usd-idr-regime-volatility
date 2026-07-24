# USD/IDR Regime-Aware Volatility Forecasting

Proyek course-project ini mengembangkan ulang forecasting USD/IDR dengan tiga pertanyaan yang sengaja dipisahkan:

1. Kapan pasar berada pada rezim volatilitas tinggi?
2. Seberapa baik keluarga GARCH meramalkan varians return hari berikutnya?
3. Seberapa baik fitur historis yang tersedia saat ini mengklasifikasikan arah USD/IDR berikutnya?

Ini adalah artefak pembelajaran dan portfolio, bukan sinyal trading, rekomendasi investasi, ataupun bukti kausal bahwa sebuah peristiwa menyebabkan perubahan kurs.

## Hasil eksekusi snapshot

Snapshot Yahoo Finance yang tersimpan mencakup **1 Juli 2016–24 Juli 2026** (2.618 observasi harga USD/IDR; 2.597 observasi setelah pembentukan fitur). Train berakhir pada 19 Juli 2024; 520 hari observasi berikutnya menjadi test set final. Seluruh angka di bawah berasal dari eksekusi kode yang tersimpan, bukan placeholder.

| Tugas | Model | Hasil test set |
| --- | --- | ---: |
| Forecast varians | GARCH(1,1) | QLIKE **-0,0405**; MAE varians 0,8396; RMSE varians 3,0978 |
| Forecast varians | EGARCH(1,1) | QLIKE -0,0076; MAE varians **0,7240**; RMSE varians **3,0380** |
| Forecast varians | GJR-GARCH(1,1) | QLIKE -0,0375; MAE varians 0,8408; RMSE varians 3,1072 |
| Arah USD/IDR | Persistence-sign baseline | 49,23% akurasi |
| Arah USD/IDR | Logistic dengan fitur lag, volatilitas, US10Y, dan probabilitas regime terfilter | **54,04%** akurasi |

Tidak ada “pemenang” tunggal pada forecasting volatilitas: GARCH memiliki QLIKE terbaik (lebih rendah lebih baik), sedangkan EGARCH memiliki MAE/RMSE varians paling rendah. Untuk arah, kenaikan terhadap baseline kecil; jangan menafsirkannya sebagai prediktabilitas ekonomi yang kuat.

Regime high-volatility yang diestimasi secara endogen memiliki rata-rata absolute return 1,177% dibanding 0,275% pada kondisi low-volatility probability. Itu adalah diagnostic yang konsisten dengan definisi rezim, bukan validasi kausal atas timeline peristiwa.

## Metodologi

### 1. Data dan informasi yang tersedia

- `IDR=X`: rate IDR per USD dari Yahoo Finance melalui `yfinance`.
- `^TNX`: proksi yield US Treasury 10Y; nilai terakhir yang dipublikasikan diteruskan hanya pada hari pasar AS tutup.
- `data/raw/manifest.json` menyimpan rentang, waktu pengambilan, ticker, dan SHA-256 snapshot.
- [Bank Indonesia BI-Rate](https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx) dan [inflasi](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx) adalah sumber resmi yang direkomendasikan untuk ekstensi makro. Mereka belum dipakai pada hasil utama karena snapshot historis terstruktur belum dapat ditarik dengan endpoint publik yang stabil pada saat reproduksi.

Proyek tidak memakai ticker Yahoo yang tidak tervalidasi untuk yield Indonesia 10Y. Kontrak input untuk `BI-Rate`, inflasi YoY, dan `ID10Y` tersedia di [data/raw/README.md](data/raw/README.md); jika ketiganya disediakan, penyelarasan dilakukan **as-of backward** berdasarkan tanggal ketika data telah diketahui pasar.

### 2. Hamilton (1989) Markov switching

`statsmodels.tsa.regime_switching.MarkovRegression` diestimasi pada log return harian (%) dengan dua state, intercept yang dapat berbeda, dan `switching_variance=True`.

Untuk menjaga notebook dapat dijalankan ulang dalam waktu wajar, optimisasi memakai tiga inisialisasi acak dengan seed 42 dan lima iterasi EM awal. Ini adalah trade-off reproducibility/runtime, bukan pencarian global yang sempurna.

- Label high-vol bukan nomor regime bawaan; ia dipilih dari realized variance state yang lebih tinggi.
- Grafik overlay peristiwa memakai probabilitas **smoothed** untuk pembacaan retrospektif saja.
- Model arah menggunakan probabilitas **filtered** yang diestimasi dengan parameter dari train set dan kemudian dilag satu hari. Ini mencegah future return masuk sebagai fitur.

Timeline perang dagang Maret 2018, pandemi Maret 2020, dan siklus kenaikan The Fed Maret 2022 hanya dipakai sebagai annotation ekonomi—bukan label target atau bukti validitas model.

### 3. GARCH-family dan evaluasi

Semua model menggunakan return harian dalam persen dan inovasi Student-t:

- GARCH(1,1),
- EGARCH(1,1),
- GJR-GARCH(1,1).

Forecast dilakukan one-step-ahead di test set. Parameter direfit setiap 63 observasi (sekitar satu kuartal hari bursa) menggunakan history yang sudah tersedia; di antara refit, varians diperbarui memakai return yang sudah teramati dan parameter terakhir. Target evaluasi adalah squared return hari berikutnya sebagai noisy proxy untuk realized variance. Karena itu metriknya adalah MAE/RMSE varians (%²) dan QLIKE—bukan directional accuracy.

Arah adalah tugas berbeda: baseline persistence-sign dibandingkan dengan Logistic Regression ber-pipeline `StandardScaler`, memakai hanya return/volatilitas/yield lag dan filtered regime probability yang tersedia sebelum target.

## Struktur

```text
data/
  raw/          # snapshot Yahoo + manifest + kontrak data makro opsional
  processed/    # fitur harian hasil pipeline
notebooks/
  01_eda_and_regime_detection.ipynb
  02_garch_forecasting.ipynb
src/
  data_ingestion.py
  feature_engineering.py
  hamilton_regime.py
  garch_modeling.py
  forecasting.py
outputs/        # CSV metrik, probabilitas regime, dan chart PNG
```

## Menjalankan ulang

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.data_ingestion --start 2016-07-01 --end 2026-07-25
MPLBACKEND=Agg .venv/bin/python -m src.forecasting
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_eda_and_regime_detection.ipynb
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/02_garch_forecasting.ipynb
```

Notebook 02 membaca snapshot yang sudah ada; tidak mengunduh ulang data. Untuk hasil yang identik dengan repository ini, gunakan file CSV snapshot yang sudah dilacak dan jangan panggil ulang data ingestion.

Jika launcher kernel Jupyter lokal tidak tersedia (misalnya sandbox desktop yang membatasi socket kernel), runner in-process yang disertakan tetap menjalankan setiap code cell secara berurutan dan menyimpan execution count:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/execute_notebook_inprocess.py notebooks/01_eda_and_regime_detection.ipynb
MPLBACKEND=Agg .venv/bin/python scripts/execute_notebook_inprocess.py notebooks/02_garch_forecasting.ipynb
```

## Artefak

- [Regime probability chart](outputs/regime_probability.png)
- [Volatility forecast chart](outputs/volatility_forecasts.png)
- [Metrik volatilitas](outputs/volatility_metrics.csv)
- [Metrik arah](outputs/direction_metrics.csv)
- [Prediksi volatilitas](outputs/volatility_forecasts.csv)
- [Probabilitas regime](outputs/regime_probabilities.csv)

## Batasan penting

- Yahoo Finance adalah sumber praktis untuk course project, bukan sumber resmi kurs acuan Bank Indonesia.
- Squared daily return adalah proxy volatilitas yang sangat berisik. Analisis lanjutan sebaiknya memakai realized volatility intraday atau horizon agregat.
- Probabilitas smoothed memakai informasi masa depan sehingga hanya cocok untuk diagnosis historis.
- Variabel makro berfrekuensi bulanan memiliki isu tanggal rilis; gunakan `available_date`, bukan tanggal observasi ekonomi.
- Event overlay menunjukkan koinsidensi waktu, bukan hubungan sebab-akibat.

Referensi metode: Hamilton, J. D. (1989), “A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle”, *Econometrica*, 57(2), 357–384. Implementasi regime mengikuti dokumentasi [statsmodels MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html); forecasting volatilitas mengikuti dokumentasi [arch](https://arch.readthedocs.io/en/stable/univariate/forecasting.html).
