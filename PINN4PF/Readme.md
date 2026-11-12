# PINN4PF: Physics-informed neural network for power flow

A shared feature network feeds two voltage-prediction branches with adaptive ReLU activations. Training combines reference-voltage error with the implemented physics penalty and its configured warmup. Validation voltage error selects the checkpoint. The packaged fresh-start configuration uses the latest large-batch budget (at most 1000 epochs); it does not require the historical warm-start weights. Optional `resume_weights` can be set explicitly if compatible weights are available.

## Use

Run from the repository root:

```bash
python PINN4PF/train.py --config PINN4PF/Results/IEEE_118/IEEE_118.json
python PINN4PF/evaluate.py --system IEEE_118
```

Training uses CUDA and saves the validation-selected checkpoint. Change the system name for the other three cases. The model is in `model.py`; shared training, losses and CSV utilities are in `Proposed/_baseline/`. This package retains the baseline's existing sample-wise split. Original server settings remain in `source_configs/`; runnable JSON files start fresh and omit unavailable warm-start paths.

## Data and outputs

Use the root `PF_DATA_ROOT` setting and the four raw CSV files described in `../Readme.md`. No data caches are used. The test set contains 4000 heatmap cases and 100 challenging cases. Results are written under `Results/IEEE_*/`. Keep `model.pt` with its configuration and dataset: evaluation checks dataset signatures. See the root README for residual definitions and the distinction between sample and grid-cell statistics.

## Method reference

Z. Kaseb, S. Orfanoudakis, P. P. Vergara and P. Palensky, "Physics-informed neural network with adaptive activation for power flow," International Journal of Electrical Power & Energy Systems, vol. 174, article 111525, 2026.
