#!/usr/bin/env bash
# Menjalankan seluruh pipeline secara berurutan (seed 42).
set -e
python -m src.train --model ae    --epochs 30
python -m src.train --model sae   --epochs 30
python -m src.train --model vae   --epochs 30
python -m src.train --model gan   --epochs 50
python -m src.train --model dcgan --epochs 20
python -m src.train --model ddpm  --epochs 12
python -m src.mlp_experiments
python -m src.evaluate
python -m src.make_figures
