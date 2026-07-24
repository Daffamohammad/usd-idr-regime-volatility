# Data snapshot dan kontrak input opsional

`yahoo_usd_idr_us10y.csv` adalah snapshot yang dipakai oleh notebook dan hasil di `outputs/`. Ticker `IDR=X` merepresentasikan IDR per USD; `^TNX` dipakai hanya sebagai proksi yield US Treasury 10Y. Detail tanggal, checksum SHA-256, dan waktu pengambilan ada di `manifest.json`.

Tiga input berikut bersifat opsional dan **tidak** dipakai pada hasil yang sudah dieksekusi. Mereka disediakan agar ekstensi makro tidak mengintroduksi data leakage:

| File | Kolom wajib | Aturan tanggal |
| --- | --- | --- |
| `bi_rate.csv` | `effective_date`, `bi_rate_pct` | tanggal keputusan/efektif yang sudah diketahui pasar |
| `inflation_yoy.csv` | `available_date`, `inflation_yoy_pct` | tanggal rilis publik, bukan akhir bulan referensi |
| `id10y_yield.csv` | `date`, `id10y_yield_pct` | tanggal observasi yield yang tervalidasi |

Semua nilai harus positif, tanggal harus unik, dan penyelarasan dilakukan dengan `merge_asof(..., direction="backward")`. Jika `id10y_yield.csv` tersedia, kode membentuk `bond_spread_pct_point = ID10Y - US10Y` serta lag satu hari untuk model arah.

Sumber yang disarankan untuk BI-Rate dan inflasi adalah halaman indikator resmi Bank Indonesia. Tidak ada ticker Yahoo yang valid untuk Indonesia 10Y dalam pemeriksaan proyek ini, sehingga seri tersebut sengaja tidak diinferensikan.
