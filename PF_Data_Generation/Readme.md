# Power flow CSV data generation

`Slover_Gen_Data.cpp` is the server generator from `Data_Gen`, including the handling that skips a failed training candidate after an `ops1`/`ops2` sparse-solver error instead of terminating the entire training-pool generation. `Structure.h`, `ThreadPool.h` and `Platform.h` supply network structures and threading support.

The generator targets 40000 valid training-source samples, a 20 × 20 heatmap with ten samples per cell, and 100 challenging N–R initial states. It perturbs injections and initial voltages, computes reference solutions and stores initial/reference states separately. Heatmap coordinates span [−2,2] on both normalized axes in steps of 0.2; the normalization uses observed training perturbation maxima. Challenging-case construction retains a valid reference solution while searching for an initial voltage that fails its N–R diagnostic. This selection diagnostic is distinct from the packaged baseline evaluator. Sample generation and validation logic are in the C++ source; the generator checks target counts before reporting completion.

## Network inputs

The repository contains `IEEE_14/`, `IEEE_118/`, `IEEE_300/` and `IEEE_1354/`. Each folder supplies `bus.txt`, `line.txt` and `gen.txt`, which the generator reads as the bus, branch and generator input tables. These network descriptions are distinct from the generated training/test CSV datasets. The provided entry point accepts all four system sizes; successful generation and sample-count validation must be checked separately for each run.

## Build and run (Linux)

Obtain NICSLU headers and the compatible shared library independently, including any license required by that distribution. The source server uses `nicslu202110`, with `linux/lib_centos6_x64_gcc482_fma/int32`. NICSLU binaries and license files are not redistributed here.

From this folder, set the include/library paths and run the wrapper:

```bash
cd PF_Data_Generation
export NICSLU_INCLUDE=/path/to/nicslu202110/include
export NICSLU_LIB=/path/to/nicslu202110/linux/lib_centos6_x64_gcc482_fma/int32
python generate.py --system 14 --output-root /absolute/path/to/new_Data
python generate.py --system 118 --output-root /absolute/path/to/new_Data
python generate.py --system 300 --output-root /absolute/path/to/new_Data
python generate.py --system 1354 --output-root /absolute/path/to/new_Data
```

The wrapper builds C++17 code, runs from this folder so network text inputs resolve correctly, and writes `Gen.log` inside the generated system directory. It refuses an existing nonempty system directory because the C++ writer appends CSV rows. Never generate over an existing dataset. Compilation does not launch training.

## Output schema

- `meta.csv`: unique sample IDs, train/test membership, difficult-case marker and perturbation coordinates.
- `bus_static.csv`, `branch_static.csv`, `ybus.csv`: network topology, bus types and admittance data.
- `bus_state.csv`: per-sample initial voltages, injections and reference voltages; angles are radians and magnitudes/powers are per unit.
- `jacobian_start.csv`, `branch_flow.csv`: additional generator diagnostics; neural loaders do not require these files.

The learning code requires `meta.csv`, `bus_static.csv`, `ybus.csv` and `bus_state.csv`. Validate generated counts, uniqueness and finite reference states before training. Training programs only read these CSV files and never write data caches.
