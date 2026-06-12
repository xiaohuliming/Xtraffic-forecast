# GWN ± ICSF/TIID Collision Experiment — Design Notes

**Goal** (audit item 2): directly test IGSTGNN's Fig-3 claim that ICSF/TIID are plug-and-play
modules that improve GWNET, on OUR clean pipeline. If no improvement → strongest hammer in the
paper ("replication failure of their core claim"). If improvement on collision windows → that
signal becomes the label-free innovation target.

## What ICSF/TIID actually do (read from baselines/IGSTGNN/src/models/IGSTGNN.py)

1. **ICSF input fusion** (`IncidentsIcsfModule.apply_incident_icsf_fusion`):
   - Embeds ONE incident per sample: position emb (12,8), desc emb (num_desc,32),
     type emb (num_types,8), holiday emb (2,4), delta_time MLP(1→16→4), + tod/dow feats
     of last history step → `incident_fusion` MLP → incident_embedding (B, d).
   - `distances` (B, N, 3) → `distance_encoder` MLP → softmax over nodes = spatial attention.
   - Q = hidden states of LAST history step; K,V = projections of incident_embedding
     expanded to N nodes, masked by distance_mask (nodes with nonzero distance features).
   - Fused attention → `incident_effect` (B, N, d); **modifies ONLY the last history
     timestep**: `enhanced[:, -1] = history[:, -1] + final_attn * V`.
2. **TIID output decay** (`_apply_incident_decay`):
   - `forecast[:, t] += incident_trans1(incident_effect) * exp(-t² / (2σ_t²))`, σ_t = 1.0
     fixed (not trained!). decay(t=1)=0.61, t=2=0.14, t=3=0.011 → only first ~2-3 steps
     of the horizon are touched.

## Inputs available in OUR batch (dist_net/data.py get_sample)

- `incident_feat` (B, M_max, C_e): dims 0..7 = OUR_TYPES one-hot
  ["1141","Fire","NoInj","UnknInj","Hazard","AHazard","CarFire","Other"], dim 8 = duration hours.
- `incident_mask` (B, M_max) bool; primary incident = first active slot.
- `rel_feat` (B, M_max, N, 4): [log_euc, log_road, up_down, _] per active event (already
  gathered; zeros for padded slots).
- `time_feat` (B, T_h, 2).

## Adapter parity (scripts/build_igstgnn_data_from_h5.py:149-211)

What the adapter fed IGSTGNN per sample, hence what a faithful port must reproduce:
- delta_time ("Event Time") = **0.0 constant**; Holiday = 0 constant; position = 0 constant.
  (So in our reproduction those three embeddings received constant inputs — keep them
  constant; they only add a learnable bias.)
- desc_code = deterministic function of type (DESC_CODE_BY_OUR_TYPE) → in OUR port desc
  embedding is redundant with type embedding; keep type only and note the deviation.
- `event_distances` (N,3) = [gaussian_kernel(log_euc), gaussian_kernel(log_road),
  (up_down>0.5)] with the closest sensor zeroed. Kernel params = adapter CLI defaults
  (sigma_euc_log / sigma_road_log / cutoff_euc_log / cutoff_road_log — read from the
  argparse defaults in the adapter before implementing).
- duration was saved but IGSTGNN's forward never consumes it.

## Port design (scripts/train_gwn_icsf.py)

GWN backbone identical to scripts/train_graphwavenet.py. Two toggleable hooks, both
copied structurally from their module, operating per-sample on the primary incident:

- `--use_icsf`: ICSFLite module — type emb (8,8) + duration MLP + distance MLP attention,
  K/V vs Q = Linear-embedded last history step; output `incident_effect` (B, N, d_icsf);
  inject by adding `proj_to_input(incident_effect)` to x_hist's LAST timestep (input space,
  3 channels) before gwnet. d_icsf = 32.
- `--use_tiid`: add `proj_out(incident_effect)[:, None, :] * exp(-t²/2σ²)` to the final
  (B, T_p, N) prediction, σ=1.0 fixed (their value).
- Samples with no active incident (none in our event-anchored sets, but guard anyway):
  zero distance_mask → effect masked to 0.

## Experiment matrix

| run | config | seeds |
|---|---|---|
| GWN (control) | already have: 12.402±0.042 Alameda, 13.480±0.077 CC | 42,1,2 |
| GWN+ICSF | --use_icsf | 42,1,2 |
| GWN+ICSF+TIID | --use_icsf --use_tiid | 42,1,2 |

Alameda + ContraCosta first (CC is where labels showed real signal in the type breakdown).
Decision read-out: compare vs GWN seed band; then per-type breakdown on collision windows
(reuse scripts/incident_type_breakdown.py pattern) — the key cell is CC collision windows
where IGSTGNN beat FDN by +1.135 affected.

## Honest-reporting notes

- Our port is faithful-in-spirit, not bit-identical (hidden-space fusion adapted to input
  space; desc dropped as constant-derived). State this in the paper.
- Their σ_t=1.0 means TIID can only affect ~3 steps; note when interpreting.
- The adapter fed delta_time=0/holiday=0/position=0 — also true for our IGSTGNN
  reproduction runs, so the collision is apples-to-apples within OUR pipeline.
