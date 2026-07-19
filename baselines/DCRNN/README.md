# Original DCRNN on XTraffic 2023

This directory runs the original DCRNN model and training recipe on XTraffic
2023. The model follows the ICLR 2018 paper and the official TensorFlow
implementation:

- paper: https://arxiv.org/abs/1707.01926
- code: https://github.com/liyaguang/DCRNN
- official configuration: `data/model/dcrnn_la.yaml` in that repository

The implementation is a faithful PyTorch port because the upstream repository
depends on TensorFlow 1.x. Model architecture, model inputs, optimizer settings,
and the learning-rate schedule are fixed in `ORIGINAL_DCRNN_CONFIG`; the command
line does not expose overrides for them.

## What stays original

| Setting | Fixed value |
|---|---:|
| history / horizon | 12 / 12 |
| encoder input | standardized traffic value + `time_in_day` |
| output | traffic value |
| DCGRU layers | 2 |
| hidden units | 64 |
| diffusion filter | dual random walk |
| maximum diffusion step | 2 |
| curriculum decay steps | 2000 |
| curriculum learning | enabled |
| batch size | 64 |
| optimizer | Adam |
| base learning rate | 0.01 |
| Adam epsilon | 0.001 |
| weight decay | 0 |
| epochs | 100 |
| LR decay epochs | 20, 30, 40, 50 |
| LR decay ratio | 0.1 |
| minimum LR | 0.000002 |
| gradient clipping | 5 |
| early-stop patience | 50 |

The decoder starts from an all-zero GO symbol, uses inverse-sigmoid scheduled
sampling during training, and is fully autoregressive during validation and
testing. The graph supports are exactly the official `dual_random_walk` pair:
`random_walk(A).T` and `random_walk(A.T).T`.

## What changes for XTraffic

Only the data adapter changes:

- METR-LA/PEMS-BAY observations are replaced by XTraffic 2023 flow;
- the original road graph is replaced by the official XTraffic weighted graph;
- XTraffic's explicit validity mask replaces the original zero-as-null rule;
- `sample_start`, split membership, node order, and affected-node evaluation are
  read from the same cache used by NDR-Net.

Incident features, affected labels, seasonal baselines, occupancy, speed, and
day-of-week are never model inputs. `affected_mask` is used only after prediction
to report grouped metrics.

The primary NDR-Net comparison uses `--protocol event`, so both models see the
same 2023 event-anchored samples and split. `--protocol full_window` is a separate
standard forecasting experiment and must not be mixed with event-protocol scores.

## HPC smoke test

Run from the repository root:

```bash
python baselines/DCRNN/train_xtraffic.py \
  --region Alameda \
  --protocol event \
  --data-dir /path/to/region_data \
  --graph-dir /path/to/region_graphs \
  --archive-dir /path/to/xtraffic_archive \
  --out-dir /path/to/dcrnn_results \
  --device cuda \
  --seed 42 \
  --smoke
```

`xtraffic_archive` must contain `adj_matrix.npy`. The HDF5 cache must have the
original 12-step history and 12-step horizon; the program stops instead of
silently changing the model when that contract is not met.

After the smoke test, remove `--smoke` for the official 100-epoch run. Use the
same seed set for DCRNN and NDR-Net. The comparison protocol currently specifies
seeds `42`, `1`, and `2`.

```bash
for region in Alameda ContraCosta Orange; do
  for seed in 42 1 2; do
    python baselines/DCRNN/train_xtraffic.py \
      --region "$region" \
      --protocol event \
      --data-dir /path/to/region_data \
      --graph-dir /path/to/region_graphs \
      --archive-dir /path/to/xtraffic_archive \
      --out-dir /path/to/dcrnn_results \
      --device cuda \
      --seed "$seed"
  done
done
```

No model or optimizer flags are accepted in the formal command. In particular,
do not add `--input-mode`, `--batch-size`, `--lr`, or `--epochs`.

## Outputs

Each run writes to:

```text
<out-dir>/<protocol>/official_graph/original_config/<region>/seed<seed>/
```

The directory contains `config.json`, `training_log.csv`, `ckpt_best.pt`,
`test_predictions.npz`, and `summary.json`. `config.json` records the fixed
official settings and sample-start fingerprints for train, validation, and test.

Report DCRNN versus NDR-Net only when region, protocol, test sample fingerprint,
target mask, horizon, and seed set agree. This result should be described as the
original DCRNN model retrained on XTraffic 2023, not as reproduction of the
METR-LA or PEMS-BAY numbers in the original paper.
