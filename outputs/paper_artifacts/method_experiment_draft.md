# Paper Draft Notes: Latent-Incident Mediated Traffic Forecasting

## Working Title

Latent-Incident Mediated Spatio-Temporal Traffic Forecasting under Incidents

## Core Problem

Conventional traffic forecasting models mainly learn regular traffic dynamics. Their error increases under incidents because the observed future state is no longer explained only by periodic and spatial traffic patterns. XTraffic also shows that inferring incident type from traffic state is unreliable, suggesting that the incident type is not the right supervision target for forecasting. We therefore model the traffic impact induced by an incident, rather than treating the incident label itself as the central prediction target.

## Method Overview

The model decomposes future traffic into a normal counterfactual component and an incident-induced residual:

```text
Delta_gated = (1 - alpha) * Delta_normal_style + alpha * Delta_incident_style
Y_hat = Y_normal + beta * Delta_gated
```

`Y_normal` is produced by a learned normal STGNN trained on normal or weak-incident traffic windows. The final model uses two residual-space branches: a normal-style residual branch and an incident graph branch. A learned gate fuses these two residual embeddings before adding the result back to the learned normal forecast. This keeps the advisor's original dual-branch gate idea, while forcing the branches to operate in incident-impact residual space rather than both predicting the full traffic state.

## Architecture

1. Normal branch: a lightweight STGNN predicts the counterfactual future traffic state under regular conditions.
2. Residual construction: both residual branches receive the learned normal residual, the statistical residual, the normalized normal-forecast disagreement `normal_delta`, and `abs(normal_delta)` as a disagreement proxy.
3. Candidate graph: for each incident, the model builds a full candidate sensor graph around the incident location instead of selecting only label-known affected top-k nodes.
4. Normal-style and incident graph branches: the normal-style branch handles mild residual corrections, while node, event, temporal, and graph features are encoded to predict incident-style residual embeddings.
5. Dual-branch gate: a learned node-horizon gate fuses the normal-style residual branch and incident graph branch in residual space.

## Experimental Story

The learned-normal residual model reduces robust MAE over the learned normal baseline. Adding `normal_delta` improves the residual branch by exposing the disagreement between the learned normal forecaster and the statistical normal reference. Dual historical residuals further improve the alignment between input residual history and future residual target. The no-aux temporal-decay model verifies that the gains are not caused by future-derived auxiliary labels. The best variant is the dual-branch gated residual no-aux model, which improves over the single residual branch by allowing normal-style and incident-style residual explanations to compete at each candidate node and forecast horizon.

## Key Results to Report

- Best model: dual-branch gated residual no-aux.
- Test all robust MAE: 0.8328 -> 0.7181, 13.78% improvement.
- Test affected robust MAE: 1.2938 -> 1.1234, 13.17% improvement.
- Across three seeds, all robust MAE is 0.7182 +/- 0.0016 and affected robust MAE is 1.1240 +/- 0.0054.
- The no-aux setting shows that the improvement does not depend on future-derived auxiliary impact labels.
- Branch interpretability: learned gate affected MAE is 1.1234, better than fixed gate 0.5 at 1.1706, normal-style residual only at 1.2478, and incident-graph residual only at 1.3562.
- Gate selection alignment: on affected elements, the mean incident-branch gate is 0.3921 when the incident branch has lower local error and 0.3511 when the normal-style branch has lower local error.
- Case studies: selected incidents where learned gate improves most over fixed gate show stronger gate weights around high residual affected nodes; sample 192208 improves affected MAE from 5.4605 to 3.9825.

## Interpretation

The improvement appears in the main full test split, affected-candidate subset, no-aux setting, and seed robustness. This matters because the comparison uses the same HDF5 cache, time split, learned normal branch, residual target, and robust MAE metric. The only structural change is the gated residual fusion, so the gain is attributable to the model's ability to choose between normal-style and incident-style residual explanations.

The gate should be interpreted as a local residual-explanation selector, not as a global incident-severity indicator. It does not monotonically increase with incident severity or recovery duration, but it does assign higher incident-branch weights when the incident branch is locally more accurate and when residual magnitude is larger on affected candidates.

## Current Caveat

Aggregate gate analysis and first case-level visualizations are now done. The remaining interpretability gap is broader qualitative coverage: include one success case, one neutral case, and one failure case if space allows.

## Recommended Figure and Table Placement

- Table 1: main forecasting results.
- Table 2: component ablation.
- Table 3: seed robustness.
- Table 4: gate and branch interpretability.
- Table 5: severity and recovery group analysis.
- Figure 1: method diagram.
- Figure 2: branch ablation for learned gate vs fixed gate and single branches.
- Figure 3: gate selection alignment.
- Figure 4: case study heatmap.
- Figure 5: horizon-wise affected-candidate MAE.
- Figure 6: severity and recovery gains from temporal decay.
