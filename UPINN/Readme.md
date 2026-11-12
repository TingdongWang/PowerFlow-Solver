# UPINN: Unsupervised physics-informed neural network

A fully connected voltage predictor is trained using the power-balance residual without a supervised voltage-target loss. The implementation uses the retained ProxRMS optimizer and learning-rate schedule. Its features contain power injections and PV voltage setpoints; it does not encode all free coordinates of the original initial voltage.

## Use

Run from the repository root:

```bash
python UPINN/train.py --config UPINN/Results/IEEE_118/IEEE_118.json
python UPINN/evaluate.py --system IEEE_118
```

Training uses CUDA and saves the validation-selected checkpoint. Change the system name for the other three cases. The model is in `model.py`; shared training, losses and CSV utilities are in `Proposed/_baseline/`. This package retains the baseline's existing sample-wise split. Original server settings remain in `source_configs/`; runnable JSON files start fresh and omit unavailable warm-start paths.

## Data and outputs

Use the root `PF_DATA_ROOT` setting and the four raw CSV files described in `../Readme.md`. No data caches are used. The test set contains 4000 heatmap cases and 100 challenging cases. Results are written under `Results/IEEE_*/`. Keep `model.pt` with its configuration and dataset: evaluation checks dataset signatures. See the root README for residual definitions and the distinction between sample and grid-cell statistics.

## Method reference

B. Jiang, C. Qin and Q. Wang, "An Unsupervised Physics-Informed Neural Network Method for AC Power Flow Calculations," IEEE Transactions on Power Systems, vol. 40, no. 5, pp. 4407–4410, 2025.
