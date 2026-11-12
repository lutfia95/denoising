# denoising

The default is the 8.24M XL model: six layers, width 256, eight heads, 32 global slots,
learned pairwise mass attention with 32 strong-peak landmarks, and a b/y complement
attention bias. Long-spectrum windows also see up to two nearest complement
partners per query within 0.05 Da, including weak remote peaks. Peak features use
the `v2` set (see `configs/train_xl.yml`). The
previous 3.16M architecture is commented in the training config for reference.

## Installation

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda and,
for GPU training, an NVIDIA driver. Tested with Python 3.12 and PyTorch 2.11 (CUDA 13.0).

### 1. Create and activate the conda environment

```bash
conda create -n denoising python=3.12 -y
conda activate denoising
```

### 2. Install PyTorch

Pick the build that matches your hardware (check your driver's CUDA version with
`nvidia-smi`, or use the selector at https://pytorch.org/get-started/locally/):

```bash
# NVIDIA GPU, recent driver (CUDA 13.0)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# NVIDIA GPU, older driver (CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the remaining packages

```bash
pip install -r requirements.txt
```

`autogluon` is only needed for `src/training/train_autogluon.py`; install it separately
with `pip install autogluon.tabular` if you use that script.

### 4. Check the installation

```bash
python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -m pytest tests -q
```

The same commands work in Windows PowerShell and in WSL/Linux. Run everything from the
repository root.

## Usage

```bash
python split_data_xl.py --config config/train_xl.yml
python -u -m src.training.train_xl --config config/train_xl.yml
python -u -m src.inference.run_xl --config config/infer_xl.yml
```

Edit the canonical settings in `configs/train_xl.yml` and the input MGF paths in
`configs/infer_xl.yml`. Split directories and training output directories must be
new for fresh experiments. Existing XL splits can be reused for training.

## Hardware

Before complement-neighbor connections were added, training used about 1.5 GiB peak, including
the longest spectrum (~37k peaks). Measured on an RTX 4070 Laptop GPU (8 GB): about
18,000 training peaks/s, roughly 6 hours per epoch on the current split (369M training
peaks). These measurements need repeating with complement neighbors enabled;
the extra attention keys increase compute and workspace without adding parameters.


┌────────────────────┬─────────┬─────────────┬──────────────┬──────────┐
│        Set         │ Spectra │    Peaks    │ Signal peaks │ Signal % │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ Train              │ 270,577 │ 368,994,307 │ 41,005,880   │ 11.11    │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ ├ annotations_1pct │ 238,259 │ 344,153,946 │ 38,684,577   │ 11.24    │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ └ legacy_5pct      │ 32,318  │ 24,840,361  │ 2,321,303    │ 9.34     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ Val                │ 60,031  │ 160,508,881 │ 11,324,189   │ 7.06     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ ├ annotations_1pct │ 54,970  │ 158,711,118 │ 11,017,968   │ 6.94     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ └ legacy_5pct      │ 5,061   │ 1,797,763   │ 306,221      │ 17.03    │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ Test               │ 62,643  │ 222,718,490 │ 11,997,116   │ 5.39     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ ├ annotations_1pct │ 54,579  │ 213,668,992 │ 11,323,498   │ 5.30     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ └ legacy_5pct      │ 8,064   │ 9,049,498   │ 673,618      │ 7.44     │
├────────────────────┼─────────┼─────────────┼──────────────┼──────────┤
│ Total              │ 393,251 │ 752,221,678 │ 64,327,185   │ 8.55     │
└────────────────────┴─────────┴─────────────┴──────────────┴──────────┘
