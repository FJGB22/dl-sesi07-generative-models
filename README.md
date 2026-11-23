# Studi Komparatif Empiris Model Generatif — Fashion-MNIST

Tugas Take-Home Deep Learning (D6421) Sesi 07 — **Jonathan Kuniardi (2802461832), kelas LA01**

Repositori: https://github.com/FJGB22/dl-sesi07-generative-models

Implementasi PyTorch dan perbandingan terstandarisasi enam model generatif — **Vanilla AE, Sparse AE, VAE,
Vanilla GAN (MLP), DCGAN, DDPM** — serta eksperimen pembuktian bahwa bobot dan fitur laten **MLP Classifier**
tidak dapat dipakai langsung untuk rekonstruksi/generasi citra.

## Protokol
| Aspek | Nilai |
|---|---|
| Dataset | Fashion-MNIST (28×28 grayscale, 10 kelas), 70.000 citra digabung |
| Split | 56.000 train / 7.000 val / 7.000 test (80/10/10), `torch.Generator().manual_seed(42)` |
| Seed | `random`, `numpy`, `torch` = 42 (`src/utils.py::set_seed`) |
| Batch | 128, `shuffle=True` (train), `shuffle=False` (val/test) |
| Metrik | Test MSE, PSNR, SSIM (`skimage`), FID (InceptionV3 `pytorch-fid`, 2.000 sampel vs 7.000 citra test), latensi batch 64 |

## Struktur repositori
```
src/
  utils.py            seed, device, path output, helper JSON
  data.py             loading Fashion-MNIST + split 80/10/10 ber-seed
  metrics.py          MSE / PSNR / SSIM / FID / latensi
  models/
    autoencoders.py   VanillaAE, SparseAE (L1), VAE (reparameterization)
    gans.py           MLP GAN, DCGAN (+ inisialisasi N(0, 0.02))
    ddpm.py           U-Net kecil + jadwal beta linier, T = 1000
  train.py            training pipeline semua model generatif
  mlp_experiments.py  Bagian 2: Tahap 1–5 batasan MLP Classifier
  sample_ddpm.py      sampling reverse diffusion untuk FID & trajectory
  evaluate.py         tabel metrik -> outputs/results/metrics.json
  make_figures.py     Figur 0–5 -> outputs/figures/
  anomaly_proxy.py    eksperimen pendukung Q6: deteksi cacat sintetis via reconstruction error
notebooks/colab_run.ipynb   menjalankan seluruh pipeline di Google Colab (GPU)
run_all.sh                  pipeline lengkap berurutan
report/                     sumber LaTeX laporan
outputs/results/            semua angka hasil eksperimen (JSON)
outputs/figures/            figur laporan (300 dpi)
outputs/logs/               log training
```

## Cara menjalankan
```bash
pip install -r requirements.txt
bash run_all.sh            # atau jalankan langkah per langkah di bawah
```
Langkah per langkah:
```bash
python -m src.train --model ae    --epochs 30
python -m src.train --model sae   --epochs 30
python -m src.train --model vae   --epochs 30
python -m src.mlp_experiments
python -m src.train --model gan   --epochs 50
python -m src.train --model dcgan --epochs 20
python -m src.train --model ddpm  --epochs 12
python -m src.sample_ddpm --n 2000
python -m src.evaluate
python -m src.anomaly_proxy
python -m src.make_figures
python report/build_numbers.py   # angka laporan dari JSON, lalu kompilasi report/main.tex
```
Dataset diunduh otomatis lewat `torchvision.datasets.FashionMNIST` ke `data/`. GPU dipakai otomatis bila tersedia;
pada CPU, argumen `--threads N` membatasi jumlah thread. Di Google Colab, buka `notebooks/colab_run.ipynb`
dan pilih runtime T4 GPU.

Hasil yang dilaporkan dijalankan pada CPU 2-core (tanpa GPU); total waktu ±3 jam, didominasi training dan
sampling DDPM 1000 langkah.

## Hasil ringkas (Test Set, CPU 2-core)
| Model | Train/epoch (s) | Latensi batch 64 (ms) | MSE | PSNR (dB) | SSIM | FID (2k sampel) |
|---|---|---|---|---|---|---|
| Vanilla AE | 2.7 | 0.37 | 0.0065 | 22.98 | 0.829 | – |
| Sparse AE | 3.0 | 0.36 | 0.0097 | 21.12 | 0.762 | – |
| VAE | 4.0 | 0.18 | 0.0132 | 19.64 | 0.708 | 89.2 |
| Vanilla GAN | 14.6 | 1.20 | – | – | – | 124.6 |
| DCGAN | 182.0 | 22.19 | – | – | – | 31.9 |
| DDPM | 478.5 | 93820.89 | – | – | – | 28.0 |

FID citra riil (val vs test) = 8.4. MLP Classifier: akurasi test 88.93%;
frozen-latent decoder PSNR 18.31 dB vs Vanilla AE 22.98 dB;
sampling ruang laten MLP FID 149.1.

Angka lengkap ada di `outputs/results/metrics.json` dan `outputs/results/mlp_experiments.json`;
ringkasan dan analisis ada di laporan PDF.
