# Method Diagram

```mermaid
flowchart LR
    X["Historical traffic state X"] --> N["Normal STGNN"]
    N --> YN["Counterfactual normal forecast Y_normal"]

    X --> RC["Residual construction"]
    YN --> RC
    STAT["Statistical normal reference"] --> RC
    RC --> F1["normal_delta"]
    RC --> F2["dual historical residual"]
    RC --> F3["abs(normal_delta)"]

    C["Incident context"] --> GC["Full candidate sensor graph"]
    GC --> IB["Incident graph residual branch"]
    F1 --> NB["Normal-style residual branch"]
    F2 --> NB
    F3 --> NB
    F1 --> IB
    F2 --> IB
    F3 --> IB
    C --> IB

    NB --> RN["Delta_normal_style"]
    IB --> RI["Delta_incident_style"]
    RN --> G["Node-horizon gate alpha"]
    RI --> G
    RC --> G
    G --> RF["Delta_gated = (1-alpha) Delta_normal_style + alpha Delta_incident_style"]
    YN --> OUT["Final forecast"]
    RF --> OUT
    OUT["Y_hat = Y_normal + beta Delta_gated"]
```

## Caption Draft

Overall architecture of the latent-incident mediated dual-branch residual-gating model. A normal STGNN first estimates the counterfactual traffic state under regular conditions. The model then constructs residual features against the learned and statistical normal references. Instead of asking two branches to predict the full traffic state from identical inputs, both branches operate in residual-impact space: a normal-style residual branch captures mild or ordinary deviations, while an incident graph branch captures spatially propagated incident impact on the full candidate graph. A node-horizon gate adaptively fuses the two residual explanations before adding the gated residual back to the normal forecast.
