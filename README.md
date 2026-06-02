# DaLC — Difficulty-aware Label Completion

Official code of *DaLC: Difficulty-aware Label Completion for Truth Inference in Crowdsourcing Truth Inference*.

## Install

Tested on Python **3.11** with PyTorch **2.8** + PyG **2.7**. Other Python 3.10+ / PyTorch 2.1+ combinations should work but pin the PyG extension wheels to your `torch` build.

```bash
# 1. Create env
conda create -n dalc python=3.11 -y && conda activate dalc
#  (or)  python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (CPU example; adjust for CUDA/MPS — see https://pytorch.org/get-started/)
pip install torch>=2.1

# 3. Install PyG + scatter/sparse wheels matching your torch build
#    (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
pip install torch_geometric torch_scatter torch_sparse

# 4. Remaining dependencies (numpy, scipy, sklearn, tensorflow, etc.)
pip install -r requirements.txt
```

**Java (required for the MV / DS / GTIC / IWBVT aggregators)** — JDK 8+ on `PATH` or in `JAVA_HOME`.

```bash
# macOS (Homebrew)
brew install openjdk
echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc

# Linux — Debian / Ubuntu
sudo apt-get install -y default-jdk

# Linux — Fedora / RHEL / CentOS
sudo dnf install -y java-17-openjdk-devel

# Verify
java -version && javac -version
```

The pre-compiled CEKA `.class` files ship under `examples/Ceka-v1.0.1/build/`; the wrapper recompiles on demand if anything is stale.

## Aggregation baselines (no completion)

Run the same 7 aggregators directly on the raw sparse votes, for the 5 real-world datasets:

```bash
python experiments/aggregation_experiment.py
```

Output: `results/_aggregation_summary/aggregation_results.csv`.

## Key DaLC parameters

| Flag | Meaning | Default |
|------|---------|---------|
| `--top-k` | neighborhood size *k* | 5 |
| `--easy-threshold` / `--ambiguous-threshold` | NCS bucket thresholds | 0.8 / 0.6 |
| `--lambda-easy / -ambiguous / -hard` | self-neighbor blend per bucket | 0.9 / 0.6 / 0.2 |
| `--epochs` | training schedule | 250 |
| `--seed` | random seed | 42 |

## Run DaLC on one dataset

```bash
python -m dalc.pipeline --dataset-dir labelme
```

Results land under `results/<dataset>/`.

## Main experiment — DaLC + 7 aggregators

Run DaLC on each of 5 real-world datasets, then apply 7 aggregators (MV, DS, GTIC, IWBVT, TiReMGE, HyperLM, CrowdFM) on top of the completed matrix:

```bash
python experiments/main_experiment.py
```

Output: `results/_main_summary/main_experiment_dalc.csv`.

## Simulation experiment — DaLC on 34 simulated datasets

Run DaLC + MV on the 34 CEKA-simulated crowdsourcing datasets (paper Table 4, DaLC column):

```bash
python experiments/simulation_experiment.py
```

Output: `results/_simulation_summary/simulation_experiment_dalc.csv` — one row per dataset with Easy / Ambiguous / Hard / overall accuracy.

