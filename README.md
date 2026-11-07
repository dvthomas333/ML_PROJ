# CIFAKE Robustness: Spatial, Frequency-Domain and Hybrid CNNs

This repository contains the implementation for an ML project evaluating lightweight convolutional neural networks for distinguishing real images from AI-generated images in the CIFAKE dataset, including evaluation under image post-processing.

The main comparison is between three representations/models:

- **S-CNN (spatial):** learns directly from RGB image pixels.
- **F-CNN (DCT):** uses a frequency-domain representation based on the discrete cosine transform.
- **H-CNN (hybrid):** combines spatial and frequency-domain branches.

The code also includes FFT and block-DCT representation variants, feature-selection analysis, perturbation-aware training, robustness evaluation, statistical comparisons, and model interpretation utilities.

## Repository structure

```text
ML_PROJ/
├── configs/
│   └── experiment.yaml       # Dataset paths and experiment settings
├── scripts/
│   └── run_all.sh            # Full experimental workflow
├── src/
│   ├── data.py               # Dataset decoding, audit, de-duplication and split
│   ├── evaluate.py           # Evaluation on clean and perturbed test data
│   ├── interpret.py          # Model interpretation / attribution analysis
│   ├── metrics.py            # Evaluation metrics
│   ├── models.py             # Spatial, frequency and hybrid CNN architectures
│   ├── perturbations.py      # Image post-processing and perturbation conditions
│   ├── representations.py    # Spatial, DCT, FFT and block-DCT representations
│   ├── spectral_analysis.py  # Spectral statistics and feature ranking
│   ├── stats.py              # Paired statistical comparisons and correction
│   ├── train.py              # Model training
│   ├── tune.py               # Hyperparameter tuning
│   └── utils.py              # Shared configuration and utility functions
├── README.md
└── requirements.txt
```

## Dataset

The experiments use the **CIFAKE – Real and AI-Generated Synthetic Images** dataset.

- Original dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The original dataset is expected to have this directory layout:

```text
data/
└── cifake/
    ├── train/
    │   ├── REAL/
    │   └── FAKE/
    └── test/
        ├── REAL/
        └── FAKE/
```

By default, `configs/experiment.yaml` sets `data.root` to `data/cifake`. If the dataset is stored elsewhere, update that setting before running the pipeline.

The dataset is not stored in this Git repository. Download it separately from Kaggle. The preprocessing step creates `data/processed/cifake_clean.npz` and `data/processed/audit.json` using the configured paths.

## Environment setup

The pinned package versions are listed in `requirements.txt`. Python 3.11 is recommended.

Create and activate a virtual environment, then install the dependencies:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS / Kaggle**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for the full training and tuning workflow. The configured training settings use mixed precision (`amp: true`). For CPU-only or memory-constrained environments, adjust the training settings in `configs/experiment.yaml` as appropriate.

## Running the project

Run commands from the repository root.

### 1. Prepare and audit the dataset

```bash
python -m src.data --config configs/experiment.yaml
```

This decodes the source images, checks image dimensions and storage metadata, removes exact duplicates and train/test exact-image overlap, creates a stratified validation split from the training data, and writes the processed arrays and audit report to the configured locations.

### 2. Run spectral analysis

```bash
python -m src.spectral_analysis
```

### 3. Run hyperparameter tuning

```bash
for model in spatial dct fft blockdct hybrid; do
  python -m src.tune --model "$model"
done
```

The tuning step writes model-specific parameter files under `configs/tuned/`. Run it before the main training stage because the training script uses those files.

### 4. Train models and evaluate

To execute the complete workflow in the order defined by the project script:

```bash
bash scripts/run_all.sh
```

The script runs data preparation, spectral analysis, tuning, the main spatial/DCT/hybrid experiments, representation and feature-selection ablations, robustness evaluation, efficiency evaluation, statistical comparisons, and interpretation.

`run_all.sh` is a full research workflow rather than a quick smoke test. It launches multiple training runs and tuning trials and can require several GPU-hours. It should be run only after the dataset and environment are prepared.

## Configuration

Edit `configs/experiment.yaml` to change:

- Dataset root and generated preprocessing/audit paths
- Image representation settings
- Model widths and dropout
- Training epochs, batch size, learning rate and worker count
- Perturbation and augmentation ranges
- Evaluation seeds, bootstrap settings and cache location
- Hyperparameter tuning trials and analysis subsample sizes

Keep the configuration used for a reported experiment with the corresponding results so the run can be reproduced.

## Generated outputs

The workflow creates generated files and directories such as:

- `data/processed/` – processed dataset arrays and audit report
- `data/perturbed/` – cached perturbed test images
- `configs/tuned/` – tuned hyperparameter files
- `runs/` – model checkpoints and per-run outputs
- `results/` and `masks/` – evaluation, analysis and feature-selection outputs

These are generated during execution and are not included in this source-code repository. Ensure sufficient disk space is available; the perturbation cache and model outputs can be large.

## Notes on reproducibility

- The experiment settings are centralised in `configs/experiment.yaml`.
- The configured evaluation seeds are `0` through `4`.
- The validation split is stratified and controlled by `split_seed`.
- The perturbation cache uses the configured `perturb_seed`.
- Record the exact dataset version, configuration, software environment and code revision used when producing results.

This repository provides the implementation and run instructions. Actual results should be taken from the outputs produced by executing the code with the specified dataset and configuration; this README does not claim that a fresh end-to-end run has been performed from this repository.
