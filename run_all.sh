#!/usr/bin/env bash
# Menjalankan seluruh pipeline secara berurutan (seed 42). GPU dipakai otomatis jika tersedia.
set -e
python -m src.train --model ae    --epochs 30
python -m src.train --model sae   --epochs 30
python -m src.train --model vae   --epochs 30
python -m src.mlp_experiments          # Bagian 2 (butuh ae.pt & vae.pt)
python -m src.train --model gan   --epochs 50
python -m src.train --model dcgan --epochs 20
python -m src.train --model ddpm  --epochs 12
python -m src.sample_ddpm --n 2000     # 1000 langkah reverse diffusion
python -m src.evaluate                 # tabel metrik -> outputs/results/metrics.json
python -m src.anomaly_proxy            # eksperimen pendukung Q6(1)
python -m src.make_figures             # Figur 0-5 -> outputs/figures/
