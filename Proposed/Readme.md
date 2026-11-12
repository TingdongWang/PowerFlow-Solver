# Proposed: voltage-state-input neural power flow solver

The initializer takes four features per non-slack bus: normalized specified active power, masked reactive power, original voltage angle and original voltage magnitude. `include_state: true` is enabled in all four runnable configurations. For N buses, its input dimension is 4(N−1) and its output dimension is 2(N−1). The evaluated MLP has two hidden layers of width 1536, SiLU activations and layer normalization after the first activation.

Offline training fits a linear voltage–power increment mapping from symmetric perturbations around training reference solutions, constructs the correction weights offline, and trains the MLP against reference voltages. During solution, the MLP refines the supplied state; the correction layer repeatedly maps recomputed nonlinear power mismatch to clipped voltage updates. Configured momentum is applied during correction. Online iterations do not construct a Jacobian or solve a linear system. PV magnitudes remain prescribed.

## Use

From the repository root:

```bash
python Proposed/main.py --config Proposed/Results/IEEE_118/IEEE_118.json
```

`mode` accepts `prepare` (CSV validation), `check` (preflight), `train_test` and `test`. Test mode requires the corresponding `Results/IEEE_118/model.pt`. `inference.py` provides `load_solver` and `solve` for independent use: `initial` and `spec` have shape `[batch, N−1, 2]`, with `[angle, magnitude]` and `[P, Q]` respectively, in checkpoint bus order. `solve` returns voltages, maximum residuals and neural-call counts.

The grouped split keeps identical effective injections together and checks train/test injection overlap. Group-level splitting can slightly change the requested 36000/4000 counts. The saved split audit gives the actual counts and sample IDs. Momentum candidates and budgets are copied from the voltage-input experiment, including zero momentum where configured; they are not silently changed to match the earlier power-only experiment.

## Data and outputs

Use the root `PF_DATA_ROOT` setting and the four raw CSV files described in `../Readme.md`. No data caches are used. The test set contains 4000 heatmap cases and 100 challenging cases. Results are written under `Results/IEEE_*/`. Keep `model.pt` with its configuration and dataset: evaluation checks dataset signatures. See the root README for residual definitions and the distinction between sample and grid-cell statistics.
