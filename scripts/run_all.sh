#!/usr/bin/env bash
# Full experimental protocol. Run from the repository root after placing CIFAKE in data/cifake/.
set -euo pipefail
SEEDS="0 1 2 3 4"

# 1. Data audit, de-duplication, split
python -m src.data

# 2. Model-agnostic spectral analysis + MI feature ranking (writes masks/)
python -m src.spectral_analysis

# 3. Equal-budget hyper-parameter search (clean validation AUC)
for m in spatial dct fft blockdct hybrid; do python -m src.tune --model "$m"; done

# 4. Main experiment: 3 models x 2 regimes x 5 seeds = 30 runs
for m in spatial dct hybrid; do
  for r in clean augmented; do
    for s in $SEEDS; do python -m src.train --model "$m" --regime "$r" --seed "$s" --params "configs/tuned/$m.json"; done
  done
done

# 5. Representation ablation: FFT and 8x8 block DCT, clean regime, 5 seeds = 10 runs
for m in fft blockdct; do
  for s in $SEEDS; do python -m src.train --model "$m" --regime clean --seed "$s" --params "configs/tuned/$m.json"; done
done

# 6. Feature-selection ablation: DCT model on top-k MI coefficients, 3 seeds x 3 k = 9 runs
for k in 256 768 1536; do
  for s in 0 1 2; do
    python -m src.train --model dct --regime clean --seed "$s" --params configs/tuned/dct.json --mask "masks/mi_top$k.npy"
  done
done

# 7. Robustness evaluation, efficiency, statistics, attribution
python -m src.evaluate
python -m src.evaluate --efficiency
for r in clean augmented; do
  python -m src.stats --a dct --b spatial --regime "$r"
  python -m src.stats --a hybrid --b spatial --regime "$r"
  python -m src.stats --a hybrid --b dct --regime "$r"
done
python -m src.interpret --spatial-run runs/spatial_clean_s0 --freq-run runs/dct_clean_s0
