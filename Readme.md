# Efficient Batch AC Power Flow with Tight Power-Mismatch Control via Offline-Learned Linear Correction and Residual Feedback

Source code accompanying the manuscript **Efficient Batch AC Power Flow with Tight Power-Mismatch Control via Offline-Learned Linear Correction and Residual Feedback** by Tingdong Wang, Wenzheng Ma, Yihu Liu, and Yinglong Ma. The repository contains the proposed solver, local implementations of the GIN, PINN4PF, and UPINN baselines, and the CSV data generator.

Repository: https://github.com/TingdongWang/PowerFlow-Solver.

## Method overview

The proposed batch AC power flow (AC-PF) solver combines learned voltage initialization, an offline-learned linear correction operator, and nonlinear residual feedback:

- **Learned initialization:** An MLP uses specified power injections and supplied initial voltage states to estimate the starting voltage.
- **Offline-learned linear correction:** Symmetric perturbations across training operating conditions provide voltage--power increment pairs. Regularized least squares fits a shared forward mapping, whose inverse forms a fixed, bias-free correction operator.
- **Residual feedback:** The solver recomputes the nonlinear AC power mismatch after each voltage update and reuses the correction operator, avoiding online Jacobian construction and linear-system solves.

The manuscript evaluates IEEE 14-, 118-, and 300-bus systems and the PEGASE 1354-bus system under in-distribution and out-of-distribution (OOD) perturbation conditions, as well as on selected Newton–Raphson (N–R) power flow divergence cases. The results demonstrate tight power-mismatch control across the tested perturbation grid and efficient GPU batch processing. Voltage-state accuracy, ablations, and correction strategies with shared MLP initialization further assess the solver. The folder name `IEEE_1354` refers to the PEGASE 1354-bus case.

## Repository structure

```text
Github/
├── Proposed/                 # MLP(P, Q, original angle, original magnitude) + residual correction
│   ├── main.py, model.py, train.py, inference.py
│   ├── data.py, Function.py, evaluate.py, runtime.py
│   ├── _baseline/            # Shared baseline training, CSV loading and evaluation utilities
│   ├── Results/IEEE_*/IEEE_*.json
│   └── source_configs/       # Original server configurations for provenance
├── GIN/                      # Graph iteration network
├── PINN4PF/                  # Physics-informed network with adaptive activations
├── UPINN/                    # Unsupervised physics-informed network
├── PF_Data_Generation/       # C++ generator and 14/118/300/1354 network input files
├── requirements.txt
├── environment.json
└── source_manifest.json      # Source and packaged file hashes
```

Each method has its own `Readme.md` and configurations for IEEE_14, IEEE_118, IEEE_300 and IEEE_1354. Run the commands below from this repository root on Linux. Network input text files for all four systems are included in `PF_Data_Generation/IEEE_*/`. Generated sample CSV datasets, trained weights, historical logs and proprietary solver libraries are not bundled.

## Hardware and software

The source experiments use Ubuntu 26.04 LTS, two Intel Xeon Gold 6526Y processors (32 physical cores / 64 logical CPUs in total), approximately 247 GiB of OS-reported RAM, and one NVIDIA GeForce RTX 5090 (32607 MiB reported VRAM). The NVIDIA driver is 595.84. Neural training uses the GPU.

The measured Python environment is Python 3.10.20, PyTorch 2.11.0+cu128 (CUDA runtime 12.8), NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 and Matplotlib 3.9.2. The generator is built with GCC/G++ 15.2.0 and C++17. CUDA runtime 12.8 refers to the PyTorch build, not the maximum CUDA version displayed by the driver. See `environment.json` for the captured versions.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Linux is required by the current file-locking code (`fcntl`).

## Data preparation

The four bundled network cases each contain `bus.txt`, `line.txt` and `gen.txt`. These are generator inputs, not the generated training/test CSV samples. Use `PF_Data_Generation/generate.py` to produce the sample datasets; see that folder's README for dependencies and commands.

Set `PF_DATA_ROOT` to the directory containing the four system folders. The default is a `Data` directory alongside this repository. This environment variable controls the CSV location for every method; the `data_root` value in the JSON files documents the default layout.

```bash
export PF_DATA_ROOT=/absolute/path/to/Data
# Required for each system:
# Data/IEEE_118/{meta.csv,bus_static.csv,ybus.csv,bus_state.csv}
```

The CSV files supply sample membership, network parameters, original initial voltages, specified injections and reference voltages. Angles are in radians and voltages/powers are in per unit. Loaders read raw CSV only and do not create or consume processed tensor caches. The expected pool contains 40,000 in-distribution development samples and 4,100 test samples: 4,000 heatmap samples (20 × 20 cells, ten per cell) and 100 selected N–R power flow divergence cases. Only in-distribution samples enter the training/validation split; OOD samples are reserved exclusively for testing. Seed 42 is used for splitting.

For the proposed method and the learning-based baselines, samples are grouped by effective injections and split with seed 42 into 36,000/4,000 training/validation samples. Test membership follows the CSV metadata. Refer to `PF_Data_Generation/Readme.md` for generation and available network inputs.

The heatmap coordinates $\alpha_{\rm ini}$ and $\alpha_{\rm inj}$ denote signed normalized perturbation levels of initial voltage states and power injections, respectively, scaled by the corresponding observed maxima in the in-distribution development set. The test grid covers $[-2,2]^2$, and the in-distribution envelope is $[-1,1]^2$. OOD perturbation conditions have at least one coordinate in $[-2,-1)\cup(1,2]$.

The 100 selected N–R power flow divergence cases per system have convergent reference solutions from base-voltage initializations. Their perturbed initializations cause residual growth or solver failure in N–R; slow convergence alone does not qualify a case.

## Running the methods

```bash
python Proposed/main.py --config Proposed/Results/IEEE_118/IEEE_118.json
python GIN/train.py --config GIN/Results/IEEE_118/IEEE_118.json
python GIN/evaluate.py --system IEEE_118
python PINN4PF/train.py --config PINN4PF/Results/IEEE_118/IEEE_118.json
python PINN4PF/evaluate.py --system IEEE_118
python UPINN/train.py --config UPINN/Results/IEEE_118/IEEE_118.json
python UPINN/evaluate.py --system IEEE_118
```

Replace `118` with `14`, `300` or `1354` for another system. Proposed trains and tests in one command. To evaluate its saved model only, set `mode` to `test` in its JSON and rerun the command. Baseline training and evaluation are separate. Batch sizes and training budgets are in each JSON; reduce batch sizes for smaller GPUs. These direct entry points stop on failure; they do not launch the server's automatic OOM retry or monitoring queues.

The `Results/IEEE_*/IEEE_*.json` files are portable fresh-start configurations. Original server configurations are preserved under `source_configs/`, including any historical warm-start paths. GIN and PINN4PF server runs can have multiple training stages; their predecessor weights are not included. Fresh-start runs therefore do not reconstruct those historical checkpoint trajectories. Existing source configurations are provenance, not executable commands on another machine.

## Outputs and evaluation

Each method writes logs, `model.pt` where applicable, `summary.json`, `heatmap.csv` and `ill.csv` under its own `Results/IEEE_*/` directory. Existing outputs at the same location can be replaced by another run. Keep separate copies for separate experiments.

For each sample, `max_pf_mismatch = max(max_abs_p_mismatch, max_abs_q_mismatch)`, using active mismatches at non-slack buses and reactive mismatches at PQ buses. The strict convergence test is a finite residual smaller than 1e-4 p.u.; PV voltage error is reported separately. A heatmap cell uses the largest residual among its ten samples, not their mean. Nonfinite results remain failures. `ill.csv` retains all 100 selected N–R power flow divergence cases.

The original CSV voltage is passed to all methods. Proposed uses it as an MLP input, and GIN uses it to start its iterative updates. The direct PINN4PF and UPINN architectures use injections and specified PV magnitudes, rather than all free initial-state coordinates. Baselines are local implementations and adaptations of the cited methods, not the original authors' software distributions.
