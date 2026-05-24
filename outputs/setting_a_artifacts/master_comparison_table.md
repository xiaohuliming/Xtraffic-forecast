# Master Comparison Table — FourierDualNet vs Baselines

**Dataset**: XTraffic 2023, mainline sensors in Alameda (N=521), Contra Costa (N=496), Orange (N=990).
**Pipeline**: same h5 cache (imputed flow + event-anchored sampling), 70/15/15 time-based split, same `evaluate` function, same `y_mask_flow`.

**Setting B**: no incident labels available at inference. Evaluated on full FDN test set (each model on the same 12-step windows).

**Setting A**: incident labels available. Evaluated on the *intersection* of IGSTGNN-style and FDN-style prediction windows (both models predict the same 12-step future on these samples).

IGSTGNN was trained after we fixed a dataloader threading bug in the official code (`batch_samples[i-start_idx]` → `batch_samples[i]`); see paper §X.

## Alameda

| Setting | Model | Needs labels? | N | MAE all ↓ | MAE affected ↓ | MAE unaffected ↓ |
|---|---|:-:|---:|---:|---:|---:|
| B | GraphWaveNet | No | 9366 | 12.398 | 18.230 | 12.035 |
| B | FourierDualNet (fixed_k=3) | No | 9366 | 12.201 | 17.975 | 11.842 |
| B | FourierDualNet (learnable) | No | 9366 | **11.976** | **17.758** | **11.616** |
| A | IGSTGNN (bug-fixed) | Yes | 4561 | 13.074 | 19.194 | 12.632 |
| A | FourierDualNet (learnable, matched-window) | No | 4561 | 12.298 | 17.896 | 11.894 |

## ContraCosta

| Setting | Model | Needs labels? | N | MAE all ↓ | MAE affected ↓ | MAE unaffected ↓ |
|---|---|:-:|---:|---:|---:|---:|
| B | GraphWaveNet | No | 4040 | 13.452 | 19.770 | 13.141 |
| B | FourierDualNet (fixed_k=3) | No | 4040 | 13.167 | **19.521** | 12.854 |
| B | FourierDualNet (learnable) | No | 4040 | **13.133** | 19.568 | **12.816** |
| A | IGSTGNN (bug-fixed) | Yes | 1138 | 13.720 | 19.780 | 13.310 |
| A | FourierDualNet (learnable, matched-window) | No | 1138 | 13.748 | 19.920 | 13.331 |

## Orange

| Setting | Model | Needs labels? | N | MAE all ↓ | MAE affected ↓ | MAE unaffected ↓ |
|---|---|:-:|---:|---:|---:|---:|
| B | GraphWaveNet | No | 14368 | 13.013 | **18.206** | 12.759 |
| B | FourierDualNet (fixed_k=3) | No | 14368 | 13.020 | 18.256 | 12.764 |
| B | FourierDualNet (learnable) | No | 14368 | **12.996** | 18.284 | **12.738** |
| A | IGSTGNN (bug-fixed) | Yes | 9704 | 13.765 | 19.142 | 13.465 |
| A | FourierDualNet (learnable, matched-window) | No | 9704 | 13.296 | 18.301 | 13.016 |

## Headline takeaways

**FDN 的输入只有流量历史(每个 sample 12 步 × N 节点 × 3 通道)。它不读取**
**事故标签、事故位置、事故距离、传感器元数据等任何 IGSTGNN 使用的事故相关特征。**
Setting A / Setting B 唯一的差别是**评估窗口**,不是模型输入。

- **Setting B (label-free 部署场景)**: FourierDualNet `learnable` 持续击败单支线 GraphWaveNet baseline,Alameda + CC 上 0.32–0.42 MAE;Orange 持平。
- **Setting A (有标签场景,matched windows)**: 即使在 IGSTGNN 用全套事故标签的场景下,FDN 仍在 Alameda + Orange 上击败 IGSTGNN — overall MAE 赢 0.47–0.78,**affected MAE 赢 0.84–1.30**。Contra Costa 持平。
- **这意味着**: 在这份数据 pipeline 上,显式建模事故的 ICSF + TIID 模块并未带来比 FFT 分解 + 双 backbone 更好的预测能力,即使 FDN 看不到事故标签。
- FourierDualNet 每 batch ~10× 快于 IGSTGNN(并行架构 vs RNN-heavy)。