# GIN: Graph iteration network

Three learned Newton-type updates in rectangular voltage coordinates use global power-flow coupling. The original CSV voltage initializes the iterations. Training minimizes the sum of absolute rectangular-voltage errors; the validation residual P90 selects the checkpoint. The implementation includes Jacobian/linear-system operations, unlike the Proposed correction layer.

## Use

Run from the repository root:

```bash
python GIN/train.py --config GIN/Results/IEEE_118/IEEE_118.json
python GIN/evaluate.py --system IEEE_118
```

Training uses CUDA and saves the validation-selected checkpoint. Change the system name for the other three cases. The model is in `model.py`; shared training, losses and CSV utilities are in `Proposed/_baseline/`. This package retains the baseline's existing sample-wise split. Original server settings remain in `source_configs/`; runnable JSON files start fresh and omit unavailable warm-start paths.

## Data and outputs

Use the root `PF_DATA_ROOT` setting and the four raw CSV files described in `../Readme.md`. No data caches are used. The test set contains 4000 heatmap cases and 100 challenging cases. Results are written under `Results/IEEE_*/`. Keep `model.pt` with its configuration and dataset: evaluation checks dataset signatures. See the root README for residual definitions and the distinction between sample and grid-cell statistics.

## Method reference

J. Huang et al., "Graph Learning for Power Flow Analysis: A Global-Receptive Graph Iteration Network Method," IEEE Transactions on Network Science and Engineering, vol. 12, no. 2, pp. 599–609, 2025.
