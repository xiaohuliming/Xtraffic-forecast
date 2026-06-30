# Latent-Incident Mediated Dual-Branch Gated Traffic Forecasting under Incidents

## Abstract

Traffic incidents can abruptly alter road-network dynamics, causing traffic states to deviate from the regular spatio-temporal patterns learned by conventional forecasting models. A direct strategy is to incorporate incident type prediction or incident detection into traffic forecasting. However, evidence from XTraffic suggests that inferring incident types from traffic states is difficult, and incident categories do not necessarily reflect the actual magnitude, duration, or spatial spread of traffic disruption. This paper therefore shifts the focus from incident-type recognition to incident-impact modeling. We propose a latent-incident mediated dual-branch gated forecasting framework that decomposes future traffic states into a normal counterfactual component and an incident-induced residual. A normal STGNN first estimates the future traffic state under regular conditions. Then, two residual-space branches predict normal-style residual corrections and incident residual corrections on the full incident-centered candidate graph. A node-horizon gate adaptively fuses these two residual explanations before adding the gated residual back to the normal forecast. Experiments on XTraffic show that the lightweight no-aux dual-branch gated model reduces robust MAE from 0.8328 to 0.7181 on all candidates. Replacing the incident branch with an ST-TIS-style module further reduces all-candidate MAE to 0.7135. Freezing the residual branches and fine-tuning only the gate head further improves the best run to 0.7116 all-candidate MAE and 1.1189 affected-candidate MAE. Across three seeds, the final variant obtains \(0.7135 \pm 0.0017\) all-candidate MAE and \(1.1206 \pm 0.0029\) affected-candidate MAE, suggesting that the gain comes from gated residual impact modeling rather than future-derived auxiliary labels.

## 1. Introduction

Traffic forecasting is a central task in intelligent transportation systems, supporting route planning, congestion management, signal control, and emergency response. Modern spatio-temporal graph neural networks have achieved strong performance by modeling historical traffic dynamics over road networks. These methods typically assume that future traffic states can be inferred from recurring temporal patterns, local spatial dependencies, and recent observations. Such assumptions are effective under regular traffic conditions, but they become fragile when unexpected incidents occur.

Traffic incidents, such as accidents, road closures, adverse weather, or other disruptions, introduce external perturbations into the transportation system. After an incident occurs, flow may decrease, speed may drop, occupancy may increase, and the disruption may propagate from the incident location to upstream or downstream road segments. These changes are not merely ordinary fluctuations in the historical sequence; they are incident-induced deviations from the normal traffic regime. As a result, forecasting models trained primarily on regular traffic patterns often suffer from degraded accuracy during incident windows.

Recent datasets and models have begun to connect incident records with traffic forecasting. XTraffic provides large-scale, spatio-temporally aligned traffic and incident data, enabling post-incident forecasting and incident-related analysis \cite{gou2024xtraffic}. More recently, IGSTGNN explicitly models incident influence through spatial fusion and temporal incident-impact decay \cite{fan2026igstgnn}. These efforts highlight the value of incorporating incident information into traffic forecasting. However, a key challenge remains: incident type is not always a reliable supervision target for forecasting. The same incident category may produce weak, local, and short-lived influence in one case, but severe and long-lasting congestion in another. Conversely, different incident categories may lead to similar traffic-state deviations.

This observation motivates a different modeling objective. Instead of asking the model to identify what type of incident occurred, we ask it to learn how the incident perturbs traffic states. In other words, incident impact should be represented as a residual deviation from a normal counterfactual traffic forecast. If a normal branch can estimate what the future traffic state would have been under regular conditions, then an incident branch can focus on predicting the incident-induced residual. This decomposition naturally separates regular traffic dynamics from abnormal incident impact.

In this paper, we propose a latent-incident mediated spatio-temporal forecasting framework. The model first uses a learned normal STGNN to estimate counterfactual normal traffic states. It then constructs an incident-centered full candidate sensor graph and predicts future residual impact over all candidate nodes, rather than relying on ground-truth affected-node labels to select top-k nodes. The model receives both statistical and learned-normal residual signals, including a learned-normal disagreement feature that measures the discrepancy between the learned normal forecast and a statistical normal reference. Instead of using a single residual branch as the final architecture, the model uses a normal-style residual branch, an incident residual branch, and a node-horizon gate that fuses them in residual space. We further validate an ST-TIS-style incident branch to test whether stronger spatio-temporal fusion improves incident-impact modeling.

Our contributions are as follows.

1. We formulate incident-affected traffic forecasting as a normal-impact decomposition problem, where future traffic is modeled as a learned normal counterfactual forecast plus an incident-induced residual.
2. We design a full-candidate incident residual branch that learns residual propagation over incident-centered sensor graphs without relying on affected-node labels for candidate selection at inference time.
3. We introduce a dual-branch gated residual architecture in which normal-style and incident-style residual explanations compete at each candidate node and forecast horizon.
4. We provide empirical evidence on XTraffic showing that the no-aux gated residual model improves both all-candidate and affected-candidate forecasting and remains stable across random seeds.

## 2. Related Work

### 2.1 Spatio-Temporal Traffic Forecasting

Traffic forecasting has been extensively studied as a spatio-temporal prediction problem. Classical deep forecasting models learn temporal dependencies from historical traffic sequences, while graph-based methods further incorporate road-network structure and sensor correlations. Representative spatio-temporal graph neural networks, such as DCRNN, STGCN, and Graph WaveNet, model traffic dynamics by combining temporal sequence encoders with graph convolution or adaptive adjacency learning \cite{li2018dcrnn,yu2018stgcn,wu2019graphwavenet}. These models have substantially improved regular-condition traffic forecasting, but their primary objective remains learning recurrent and spatially correlated traffic patterns from historical observations. When incidents introduce external disturbances, the future state may deviate from these regular patterns, making pure history-driven forecasting less reliable.

Transformer-based and lightweight spatio-temporal models provide another line of progress. ST-TIS extends the Transformer with spatial-temporal information fusion and region sampling to reduce complexity while preserving joint spatial-temporal dependency learning \cite{li2023sttis}. Such models are attractive as traffic backbones because they can model complex dependencies efficiently. Our framework is complementary to these backbones: a stronger traffic forecaster can be used as the normal branch, while an ST-TIS-style module can also strengthen the incident residual branch.

### 2.2 Decomposition and Multi-Scale Time-Series Modeling

Time-series decomposition has proven useful for separating trend, seasonality, and residual variation. FEDformer combines seasonal-trend decomposition with frequency-enhanced Transformer modules for long-term forecasting \cite{zhou2022fedformer}. MSD-Mixer further emphasizes multi-scale decomposition and sub-series modeling for time-series analysis \cite{zhong2024msdmixer}. These studies show that decomposing a complex sequence into easier-to-model components can improve forecasting quality.

Our work adopts a decomposition view, but the decomposition target is different. Instead of decomposing a time series into trend and seasonality, we decompose incident-window traffic into a normal counterfactual component and an incident-induced residual. This decomposition is task-specific: the residual is not generic noise, but the traffic impact caused by an external incident. This allows the incident branch to concentrate on abnormal deviations from normal traffic dynamics.

### 2.3 Normal-State and Physics-Guided Traffic Modeling

Another related direction is traffic state estimation and normal-state modeling. Physics-informed traffic models attempt to incorporate domain knowledge, such as traffic-flow laws, into deep learning systems. Balance and Brighten, for example, studies how physical knowledge can be better released through a twin-propeller network and distillation design for traffic state estimation \cite{jiang2025balance}. Such studies suggest that traffic prediction benefits from separating data-driven patterns from structured traffic-domain constraints.

Our normal branch plays a related but distinct role. It does not explicitly solve a physics-informed estimation problem; instead, it provides a learned counterfactual reference for regular traffic. The incident branch then models the residual that cannot be explained by this normal reference. In future work, physics-informed or stronger normal-state estimators could replace the lightweight normal STGNN and further improve the counterfactual baseline.

### 2.4 Incident-Aware Traffic Forecasting

Incident-aware traffic forecasting has recently become more feasible because datasets now align incident records with traffic time series. XTraffic provides large-scale spatio-temporally aligned traffic and incident data, enabling post-incident forecasting, incident classification, and causal analysis \cite{gou2024xtraffic}. The dataset also highlights a key challenge: traffic states do not always reveal incident type accurately, and incident categories may not directly correspond to traffic impact magnitude.

IGSTGNN directly addresses incident-guided forecasting by modeling incident impact through Incident-Context Spatial Fusion and Temporal Incident Impact Decay \cite{fan2026igstgnn}. This design explicitly incorporates incident information and motivates the importance of temporal impact dissipation. Our work shares the view that incidents should be modeled as external disturbances, but differs in two ways. First, we avoid making incident type or explicit incident recognition the central supervision target. Instead, we model the latent impact as a residual from normal traffic. Second, the final model uses a node-horizon gate to choose between normal-style and incident-graph residual explanations, rather than relying on a single incident residual pathway. This makes the gate directly tied to residual correction and allows impact strength to vary across candidate nodes, horizons, and incidents.

### 2.5 Position of This Work

The proposed method connects these directions through a normal-impact decomposition. From traffic forecasting, it inherits spatio-temporal graph modeling. From decomposition-based time-series forecasting, it borrows the idea of separating complex signals into meaningful components. From incident-aware forecasting, it treats incidents as external disturbances with spatial spread and temporal persistence. The central difference is that we define the incident effect as a latent residual mediated by a learned normal counterfactual branch. This formulation lets the model improve incident-window forecasting without relying on accurate incident-type inference or ground-truth affected-node selection at inference time.

## 3. Preliminaries and Problem Formulation

Let \(G=(V,E)\) denote a road sensor graph, where each node \(v \in V\) is a traffic sensor and each edge encodes spatial proximity or road-network adjacency. At each time step \(t\), each node has a traffic state vector containing flow, occupancy, and speed. Given a historical traffic sequence \(X_{t-L+1:t}\), incident context \(c\), and an incident-centered candidate graph \(G_c=(V_c,E_c)\), the goal is to predict future traffic states:

\[
\hat{Y}_{t+1:t+H} \in \mathbb{R}^{H \times |V_c| \times C},
\]

where \(H\) is the forecasting horizon and \(C\) is the number of traffic channels.

For incident-affected forecasting, we assume that the observed future state can be decomposed into two parts:

\[
Y = Y^{normal} + \Delta^{incident},
\]

where \(Y^{normal}\) is the counterfactual traffic state under normal conditions and \(\Delta^{incident}\) is the residual impact induced by the incident. Since the true counterfactual normal state is unobserved, we estimate it with a learned normal forecaster:

\[
\hat{Y}^{normal} = f_{normal}(X_{t-L+1:t}, G).
\]

The residual module predicts two residual explanations:

\[
\hat{\Delta}^{normal} = f_{res}^{normal}(Z),
\]

\[
\hat{\Delta}^{incident} = f_{res}^{incident}(Z, c, G_c),
\]

where \(Z\) denotes residual features constructed from the history, the learned normal forecast, and the statistical normal reference. A node-horizon gate produces:

\[
\alpha = \sigma(g(Z, h^{normal}, h^{incident})).
\]

The final residual is:

\[
\hat{\Delta}
= (1-\alpha) \odot \hat{\Delta}^{normal}
+ \alpha \odot \hat{\Delta}^{incident}.
\]

The final prediction is formed by residual fusion:

\[
\hat{Y} = \hat{Y}^{normal} + \beta \cdot \hat{\Delta},
\]

where \(\beta\) is a validation-selected residual scaling coefficient and \(\odot\) denotes element-wise multiplication.

## 4. Method

### 4.1 Overview

The proposed framework consists of five components: a normal STGNN branch, a residual construction module, a normal-style residual branch, an incident graph residual branch, and a node-horizon gate. The normal branch estimates the counterfactual traffic state under regular conditions. The residual construction module compares this learned normal estimate with statistical normal references and historical observations. The two residual branches generate competing residual explanations, and the gate fuses them before the final forecast is produced.

![Method architecture](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/method_architecture.png)

This design follows a simple principle: the residual branches should not relearn ordinary traffic forecasting. Instead, they should specialize in the part of the future state that cannot be explained by normal traffic dynamics.

### 4.2 Normal Counterfactual Branch

The normal branch is trained on normal or weak-incident windows to model regular traffic dynamics. It takes historical traffic states and graph structure as input and predicts the future traffic sequence:

\[
\hat{Y}^{normal}_{t+1:t+H} = f_{normal}(X_{t-L+1:t}, G).
\]

Compared with a purely statistical normal baseline, a learned normal STGNN can capture richer temporal patterns, spatial dependencies, and region-specific traffic behavior. During incident residual training, the normal branch provides a counterfactual reference. The residual target is:

\[
\Delta^{target} = Y - \hat{Y}^{normal}.
\]

This formulation encourages the incident branch to focus on abnormal perturbations rather than full-state reconstruction.

### 4.3 Full Candidate Incident Graph

For each incident, we construct a candidate sensor graph centered around the incident location. Candidate nodes are selected according to road-region and spatial constraints such as distance, freeway direction, postmile difference, and valid sensor availability. Importantly, we do not select only ground-truth affected nodes. Instead, the residual branch is trained and evaluated on the full candidate set.

This choice is important for practical forecasting. At inference time, the model may know the incident record and approximate location, but it cannot know in advance which sensors will be affected or how far the impact will propagate. A full-candidate graph allows the model to learn both affected and unaffected responses, and it prevents the evaluation from being artificially simplified by label-based node selection.

Each candidate node is represented with node-level traffic residual features, distance and direction features, anchor indicators, graph features, and incident context features. The candidate graph supports spatial message passing among nearby sensors, allowing incident residuals to propagate across road segments.

### 4.4 Residual Features and Learned-Normal Disagreement

The residual branches use several residual signals.

First, we use the statistical historical residual, which compares recent observations against a statistical normal reference. This feature captures short-term deviations from historical normal behavior.

Second, we use the learned-normal historical residual, which compares recent observations against the learned normal branch. This aligns the historical residual inputs with the future residual target defined by the learned normal forecaster.

Third, we define a normal-forecast disagreement feature:

\[
\delta^{normal} = \frac{\hat{Y}^{normal} - \hat{Y}^{stat}}{s},
\]

where \(\hat{Y}^{stat}\) is the statistical normal forecast and \(s\) is a normalization scale. This feature exposes disagreement between two notions of normality. Large disagreement suggests that the current condition may be difficult to explain by regular traffic patterns.

Finally, we include \(|\delta^{normal}|\) as a lightweight disagreement proxy. This proxy does not require explicit uncertainty estimation, but gives the residual branches a direct signal of abnormality magnitude.

The ablation results show that these residual features are important. Adding normal disagreement improves all-candidate robust MAE from 0.7579 to 0.7434, while dual historical residuals further reduce it to 0.7254.

### 4.5 Dual Residual Branches

The normal-style residual branch predicts mild or ordinary residual corrections:

\[
\hat{\Delta}^{normal} = f_{res}^{normal}(Z).
\]

The incident graph residual branch predicts:

\[
\hat{\Delta}^{incident} = f_{res}^{incident}(Z, c, G_c),
\]

where \(Z\) denotes the constructed residual and node features, \(c\) denotes incident context, and \(G_c\) is the candidate graph.

The normal-style branch captures residuals that do not require explicit incident propagation, such as mild perturbations or normal-model correction errors. The incident graph branch encodes temporal residual patterns, node attributes, incident context, and graph structure. Spatial message passing allows nearby candidate sensors to exchange information, while temporal encoding captures recent residual evolution. The output of each branch is a node-level, horizon-level residual forecast.

### 4.6 Node-Horizon Residual Gate

The gate chooses between the two residual explanations at each candidate node and forecast horizon:

\[
\alpha_{v,h} = \sigma(g(z_v, h^{normal}_v, h^{incident}_v)).
\]

The gated residual and final forecast are:

\[
\hat{\Delta}_{v,h}
= (1-\alpha_{v,h})\hat{\Delta}^{normal}_{v,h}
+ \alpha_{v,h}\hat{\Delta}^{incident}_{v,h},
\]

\[
\hat{Y}_{v,h} = \hat{Y}^{normal}_{v,h} + \beta \cdot \hat{\Delta}_{v,h}.
\]

When \(\alpha\) is small, the model relies more on normal-style residual correction. When \(\alpha\) is large, it relies more on incident-graph residual propagation. This allows the model to suppress incident corrections on unaffected or weakly affected nodes while activating the incident branch where disruption is stronger or more persistent.

### 4.7 Training and Inference

The normal branch is first trained to forecast regular traffic patterns. Then, during residual model training, the normal branch provides counterfactual predictions for incident windows. The residual branches and gate are trained to minimize forecasting loss between the final fused prediction and the observed future state, equivalently reducing error in the residual space.

At inference time, the model requires the historical traffic sequence, incident context, and candidate graph construction around the incident location. It does not require ground-truth affected-node labels or incident impact labels to select candidate nodes. The affected labels are used only for analysis and metric reporting.

## 5. Experiments

### 5.1 Dataset

We evaluate on XTraffic, a large-scale dataset that aligns traffic sensor time series with incident records \cite{gou2024xtraffic}. The traffic channels include flow, lane occupancy, and average speed. Incident records provide spatio-temporal context for post-incident forecasting. Following the incident-centered setting, we construct candidate sensor graphs around each incident and predict future traffic states over 12 horizons.

The current experiments use three regions: Alameda, Orange, and Contra Costa. For each incident sample, the model predicts future states for all candidate nodes and reports metrics separately for all candidates, affected candidates, and unaffected candidates.

### 5.2 Evaluation Metrics

We use robust MAE in normalized residual space as the main metric. Let \(e\) denote the normalized residual prediction error. The robust MAE is computed over candidate nodes, horizons, and traffic channels after applying the same normalization used during residual training.

We report:

- all-candidate robust MAE, measuring performance over the full incident-centered candidate graph;
- affected-candidate robust MAE, measuring performance on nodes labeled as affected by incident impact analysis;
- unaffected-candidate robust MAE, measuring whether the residual branch harms nodes without clear incident impact;
- horizon-wise robust MAE, measuring how performance changes across future prediction steps.

### 5.3 Compared Variants

We compare the following variants.

1. **Statistical normal + residual STGNN.** Uses a statistical normal baseline and trains a residual STGNN on statistical residuals.
2. **Learned normal.** Uses a learned normal STGNN and trains an incident residual branch using only the future residual target.
3. **+ normal_delta.** Adds learned-normal versus statistical-normal disagreement to the incident branch.
4. **+ dual historical residual.** Adds both statistical and learned-normal historical residuals.
5. **+ disagreement proxy.** Adds \(|normal\_delta|\) as a lightweight abnormality magnitude proxy.
6. **+ temporal decay head.** Adds a horizon-wise residual modulation head to the single residual branch.
7. **Residual temporal decay no-aux.** Removes future-derived auxiliary impact labels.
8. **Dual-branch gate.** Adds normal-style and incident-graph residual branches with a node-horizon gate.
9. **Dual-branch gate no-aux.** Removes auxiliary labels from the gated residual model.
10. **ST-TIS incident branch no-aux.** Replaces the lightweight incident graph branch with temporal self-attention and graph-biased spatial attention.
11. **ST-TIS gate-head fine-tune no-aux.** Freezes the trained residual branches and fine-tunes only the gate head.

The final variant is the proposed main model.

### 5.4 Main Results

Table 1 summarizes the main forecasting results. The statistical normal residual STGNN already improves over its baseline, reducing all-candidate robust MAE from 0.8735 to 0.7378 and affected-candidate robust MAE from 1.3888 to 1.1659. This confirms that incident residual modeling is a useful direction.

Replacing the statistical normal reference with a learned normal branch lowers the baseline error from 0.8735 to 0.8328 on all candidates and from 1.3888 to 1.2938 on affected candidates. Although the first learned-normal residual branch is weaker than the statistical residual STGNN in all-candidate MAE, subsequent residual alignment features close and surpass this gap.

The best all-candidate result is achieved by the ST-TIS gate-head fine-tune no-aux model. The lightweight dual-branch gate no-aux model reduces all-candidate robust MAE from 0.8328 to 0.7181. Replacing the incident branch with the ST-TIS-style module further reduces all-candidate MAE to 0.7135. Freezing the trained residual branches and fine-tuning only the gate head further reduces all-candidate MAE to 0.7116, corresponding to a 14.56% improvement. On affected candidates, the final variant reaches 1.1189 MAE, corresponding to a 13.52% improvement. These results indicate that the gated residual structure improves not only background traffic prediction but also the nodes directly affected by incidents, without relying on future-derived auxiliary labels.

### 5.5 Component Ablation

The learned-normal ablation shows how each component contributes to the final model. Adding normal disagreement reduces all-candidate MAE from 0.7579 to 0.7434. Adding dual historical residuals further reduces all-candidate MAE to 0.7254 and affected-candidate MAE to 1.1380. This is the largest improvement among the learned-normal components, suggesting that residual-input alignment is critical.

The disagreement proxy brings a small additional improvement on all candidates, reducing MAE from 0.7254 to 0.7248. Its affected-candidate result is essentially tied with the dual-residual variant. The temporal decay head further improves affected-candidate MAE from 1.1381 to 1.1308. Removing auxiliary labels still improves performance, reaching 0.7221 all-candidate MAE and 1.1290 affected-candidate MAE. The dual-branch gate then further reduces the errors to 0.7181 and 1.1234 in the no-aux setting. The ST-TIS-style incident branch further lowers all-candidate MAE to 0.7135 and affected-candidate MAE to 1.1217. Finally, gate-head fine-tuning reaches 0.7116 and 1.1189, suggesting that stronger incident-branch modeling and later gate calibration both contribute.

### 5.6 Seed Robustness

We evaluate the lightweight dual-branch gate no-aux model, the ST-TIS-style incident-branch variant, and the final gate-head fine-tuned variant with three random seeds: 7, 11, and 23. The lightweight gate obtains \(0.7182 \pm 0.0016\) all-candidate MAE and \(1.1240 \pm 0.0054\) affected-candidate MAE. The ST-TIS-style variant improves these averages to \(0.7150 \pm 0.0013\) and \(1.1220 \pm 0.0019\), respectively. Gate-head fine-tuning further improves them to \(0.7135 \pm 0.0017\) and \(1.1206 \pm 0.0029\). The small standard deviations suggest that the observed gains are not caused by a single favorable seed.

The final model also remains stable at longer horizons. Affected-candidate robust MAE at horizon 6 is \(1.1298 \pm 0.0024\), and at horizon 12 it is \(1.3639 \pm 0.0046\).

### 5.7 Gate and Branch Interpretability

To verify that the learned gate is not merely adding parameters, we analyze the trained dual-branch gate no-aux checkpoint on the full test split with 27,499 incident samples. We do not retrain the model. Instead, we compare five inference settings under the same residual scaling coefficient: the normal baseline, the normal-style residual branch alone, the incident-graph residual branch alone, a fixed gate of 0.5, and the learned gate.

| Fusion / branch setting | All MAE | Affected MAE | Unaffected MAE |
|---|---:|---:|---:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.8057 | 1.2478 | 0.5983 |
| Incident-graph residual only | 0.9012 | 1.3562 | 0.6876 |
| Fixed gate = 0.5 | 0.7442 | 1.1706 | 0.5442 |
| Learned gate | **0.7181** | **1.1234** | **0.5279** |

![Gate branch ablation](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_branch_ablation_mae.png)

The incident-graph branch alone is not sufficient, and it performs worse than the normal-style residual branch. This shows that the incident branch is not simply a stronger predictor by itself. A fixed 0.5 gate already improves over both single branches, indicating that the two residual explanations are complementary. The learned gate further improves affected-candidate MAE from 1.1706 to 1.1234, demonstrating that the model benefits from data-dependent residual fusion rather than a constant average.

We further examine whether the gate assigns higher incident-branch weights when the incident branch is locally more accurate. Here, local accuracy is measured at each sample, forecast horizon, candidate node, and traffic channel.

| Subset | Local branch condition | Mean incident-branch gate |
|---|---|---:|
| All | incident branch has lower local error | 0.3821 |
| All | normal-style branch has lower local error | 0.3528 |
| Affected | incident branch has lower local error | 0.3921 |
| Affected | normal-style branch has lower local error | 0.3511 |
| Unaffected | incident branch has lower local error | 0.3777 |
| Unaffected | normal-style branch has lower local error | 0.3535 |

![Gate selection alignment](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_selection_alignment.png)

The learned gate assigns higher incident-branch weights when the incident branch has lower local residual error, and lower weights when the normal-style branch is locally better. This is stronger evidence than comparing affected and unaffected nodes alone, because affected labels do not fully describe where residual correction should come from. The gate acts as a local residual-explanation selector rather than a simple affected-node classifier.

The gate is also positively correlated with residual magnitude. The correlation between the gate and \(|\Delta^{target}|\) is 0.2198 over all valid elements, 0.2985 on affected candidates, and 0.0943 on unaffected candidates. Thus, the gate is more responsive to abnormal residual magnitude on affected nodes. However, the gate does not monotonically increase with incident severity or recovery duration, so we do not interpret it as a global severity indicator. Its role is better described as node-, horizon-, and channel-level residual selection.

### 5.8 Case Study Visualization

We further visualize representative test incidents to inspect the local behavior of the gate. We use two case sets. The first set contains incidents where the learned gate improves affected-candidate MAE over fixed gate = 0.5 the most. The second set is mixed: it contains success cases where the learned gate clearly improves over fixed fusion, neutral cases where the two are nearly tied, and failure cases where the learned gate underperforms fixed fusion on affected candidates. This avoids showing only favorable examples and exposes the boundary conditions of the gate.

Each case figure shows affected candidate nodes, the learned incident-branch gate, the absolute target residual, the error reduction of learned gate relative to fixed fusion, and the local error difference between normal-style and incident-graph branches.

| Rank | Sample | Region | Affected nodes | Recovery min | Learned affected MAE | Fixed affected MAE | Gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192208 | 2 | 2 | 180.0 | 3.9825 | 5.4605 | 1.4780 |
| 2 | 56226 | 0 | 2 | 180.0 | 3.8845 | 5.3330 | 1.4485 |
| 3 | 184513 | 2 | 5 | 180.0 | 3.5177 | 4.9165 | 1.3987 |
| 4 | 184542 | 2 | 5 | 180.0 | 2.6550 | 4.0505 | 1.3955 |

![Gate case study](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/case_studies/case_01_sample_192208.png)

For sample 192208, the affected sensor has much larger target residual than nearby candidates, indicating that the normal counterfactual forecast is strongly violated at that location. The learned gate assigns higher incident-branch weights around the same region, and the fixed-minus-learned error map is positive there. This suggests that the learned gate reduces error precisely where abnormal residuals are strongest, consistent with the aggregate gate-selection analysis.

The mixed case set further shows three types of gate behavior.

| Rank | Category | Sample | Region | Affected nodes | Recovery min | Learned affected MAE | Fixed affected MAE | Gain | Normal-only affected MAE | Incident-only affected MAE | Mean affected gate |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | success | 192208 | 2 | 2 | 180.0 | 3.9825 | 5.4605 | 1.4780 | 6.0167 | 5.4870 | 0.5014 |
| 2 | success | 56226 | 0 | 2 | 180.0 | 3.8845 | 5.3330 | 1.4485 | 5.2091 | 5.9619 | 0.6267 |
| 3 | neutral | 195028 | 2 | 3 | 0.0 | 0.4735 | 0.4735 | -0.0000 | 0.4866 | 0.4844 | 0.3665 |
| 4 | neutral | 187753 | 2 | 7 | 155.0 | 0.6391 | 0.6391 | 0.0000 | 0.6765 | 0.7222 | 0.4466 |
| 5 | failure | 88134 | 1 | 1 | 25.0 | 7.0882 | 4.9337 | -2.1545 | 1.9051 | 9.6011 | 0.5612 |
| 6 | failure | 60576 | 0 | 2 | 15.0 | 2.4142 | 1.0681 | -1.3461 | 2.9640 | 3.5559 | 0.4935 |

![Failure case study](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/case_studies_mixed/case_05_failure_sample_88134.png)

The success cases show that the learned gate can outperform fixed fusion when the best residual explanation varies strongly across nodes or horizons. The neutral cases show that when the two branches have similar local errors, the learned gate produces almost the same affected-candidate MAE as fixed fusion. The failure cases are especially informative. In sample 88134, the incident-only affected MAE is 9.6011, much worse than the normal-only affected MAE of 1.9051, but the learned gate still assigns a mean affected incident-branch weight of 0.5612. This causes learned fusion to underperform fixed fusion. Thus, although the gate is beneficial in aggregate, it can still over-trust the incident branch in short-recovery or single-node-impact cases where the incident branch is locally unreliable.

### 5.9 Where Does Temporal Decay Help?

To understand the role of the temporal decay head, we compare the no-decay disagreement-proxy model with the temporal-decay model under severity and recovery groups. The overall affected-candidate gain of temporal decay is 0.64%, but the gain is not uniform across incidents.

For high-severity incidents, affected-candidate MAE decreases from 1.3345 to 1.3206, giving a 1.05% gain. At horizon 12, the gain increases to 1.22%. For long-recovery incidents, affected-candidate MAE decreases from 1.2339 to 1.2224, giving a 0.94% gain, with a 1.05% gain at horizon 12.

By contrast, low-severity incidents slightly regress, with affected-candidate MAE increasing from 0.8539 to 0.8553. Mid-recovery incidents also slightly regress. This pattern is important: the temporal decay head is most helpful when incident impact is strong or persistent, which is precisely where explicit temporal impact modeling should matter most. The result supports the interpretation that the decay head captures incident persistence rather than merely adding generic model capacity.

### 5.10 Horizon-Wise Analysis

Horizon-wise results show that temporal decay consistently improves affected-candidate MAE across all 12 horizons. The gain is 0.29% at horizon 1 and increases to around 0.70% to 0.80% for several middle and later horizons. The largest affected gain is observed at horizon 9, where MAE decreases from 1.2817 to 1.2715.

This trend aligns with the motivation for temporal decay. Incident impact is not only an immediate disturbance; it evolves and dissipates over time. A horizon-wise gate can adapt the residual strength for different future steps and is therefore more useful for medium- and long-horizon incident forecasting.

## 6. Discussion

The experiments support the central claim that incident-aware traffic forecasting should model impact rather than only incident type. The normal branch provides a counterfactual reference, while the residual branches learn deviations from normal traffic. This formulation makes the forecasting task more aligned with the actual error source under incidents.

The dual-branch gate also addresses a weakness of the earlier single residual pathway. If every candidate node is forced through the same incident residual explanation, the model may over-correct weakly affected or unaffected nodes. A node-horizon gate allows mild residuals and incident-propagation residuals to compete locally, which explains why the no-aux gated model improves both all-candidate and affected-candidate MAE.

The interpretability analysis supports this explanation. The learned gate outperforms both single-branch inference and fixed 0.5 fusion, and it assigns higher incident-branch weights where the incident branch has lower local residual error. Therefore, the gate should not be described as a global incident-severity estimator. It is better understood as a local residual-explanation selector.

The mixed case study also reveals a remaining weakness. In a small number of samples, the gate overweights the incident branch even when that branch is locally less reliable. We made preliminary attempts to add branch-confidence heads, branch-error supervision, hard-example reweighting, branch loss, and oracle-style gate alignment. These variants either provided limited gains or degraded fusion. Gate-head fine-tuning partially mitigates failure case 88134, reducing its learned affected MAE from 6.8554 to 6.4817, but it still underperforms fixed fusion on that case. This suggests that future versions need better uncertainty-aware gating and branch-confidence calibration.

The grouped temporal-decay analysis remains useful as an auxiliary observation. Temporal decay is beneficial for high-severity and long-recovery incidents, but it can slightly hurt low-severity cases. This suggests that mild incidents may not require strong residual persistence modeling. Future work could combine the current dual-branch gate with recovery-conditioned temporal heads so that the model can suppress incident residuals when the estimated impact is weak.

Another limitation is that the normal counterfactual branch is still a lightweight STGNN. Stronger traffic backbones such as Graph WaveNet, D2STGNN, ST-TIS, or other state-of-the-art forecasting models could provide a better normal counterfactual branch, while the current ST-TIS-style incident branch suggests that the proposed residual-impact framework is compatible with stronger spatio-temporal modules.

## 7. Conclusion

This paper proposes a latent-incident mediated dual-branch gated framework for traffic forecasting under incidents. Rather than relying on incident type recognition, the model decomposes future traffic into a normal counterfactual forecast and an incident-induced residual. A normal-style residual branch and an incident branch produce competing residual explanations, and a node-horizon gate fuses them before residual addition. Experiments on XTraffic show that the no-aux gated model improves over single-branch and temporal-decay variants, and that replacing the incident branch with an ST-TIS-style module plus gate-head fine-tuning further improves three-seed average all-candidate and affected-candidate MAE. Gate analysis further shows that learned fusion improves over fixed averaging and aligns with locally better branch predictions. These results suggest that modeling incident impact as gated residual propagation is a promising direction for robust traffic forecasting under abnormal road conditions.
