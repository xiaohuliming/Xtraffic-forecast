# DIST-Net 设计文档

**Dual-stream Incident-aware Spatio-Temporal Network**

> 双流事故感知时空预测模型 — 用于 XTraffic 数据集上的 incident-guided traffic forecasting

---

## 0. 文档目的与版本

本文档是 **新模型架构的完整设计说明**,用于:
- 指导 PyTorch 实现
- 与导师/合作者对齐方案
- 后续论文方法章的写作依据

**版本**: v0.3 (2026-05-12),数据 pipeline + 模型代码已实现,toy run 前的最后整合。

**v0.3 changelog**(队友 review 反馈整合):
- §4 region sampling graph 改为**有向**: hub→leaf 单向 shortcut,base k-NN 仍双向。**根除 hub 节点 IN-degree 高(717→58, 12× 降幅)导致的 over-smoothing 风险**。
- §15.1 新增风险条目: 有向化后 reachability 下降(diameter 仍可控但部分 leaf↔leaf pair >4 跳),trade-off 是 over-smoothing 防御
- 实证: 数据 cache 1.65 GB(原估 30 GB,实际优化后);model code 全部 stub 替换为真实实现,parameter ~316K
- Tooling: `scripts/probe_over_smoothing.py` + `scripts/viz_region_sampling_graph.py` 用于诊断

**v0.2 changelog**(基于 review 反馈):
- §1.3 contribution 重新 frame 为 "specific combination + empirical ablation"
- §3.2 schema: 数据锚点对齐 IGSTGNN ([T+1, T+12]);affected_mask 显式标 training-only
- §3.5 新增缺失数据处理章节
- §5.2 修正 scale_long patching 冗余 + time_enc 全局 mean-pool 损失信息的 bug
- §6.5 decay 聚合改为有界 sum (tanh),明确 τ/σ 步单位
- §7 sparse attention 用 PyG MessagePassing 实现,复杂度声明改为 O(E·d)
- §9 L_normal 在 affected 节点 down-weight + 监控指标
- §11 加激活显存估算
- §12 加 toy run + buffer 改 4-5 周
- §15.3 新增训练监控指标
- §16.3 ablation 重排优先级,"合训 vs 分训" 升主表

**与现有 Codex pipeline 的关系**: 完全替代。Codex 生成的 6 阶段 `dual_branch_sttis_*` 链 + groupaware adapter 当作 baseline,新架构从零设计。

---

## 1. 设计动机

### 1.1 我们要解决什么问题

XTraffic 数据集上的 **post-incident 流量预测**: 给定发生事故后的历史流量,预测未来 1 小时(12 个 5min step)各传感器的流量、占有率、速度,**特别要求在事故影响的传感器上(affected nodes)预测准**。

### 1.2 当前 baseline 的结构性缺陷(已实证)

通过 Plan B / Plan C 实验已确认 IGSTGNN(KDD'26)在 affected raw-flow MAE 上比当前 Codex pipeline 强约 **3 MAE**。差距是结构性的,不是 loss 调优能解决:

| 缺陷 | 表现 | 新架构对策 |
|---|---|---|
| 局部 36 节点子图 | 看不见远场传播 | 全县 N=521-990 节点 + region sampling |
| z-residual 范式 | 信息在归一化时损失 | 直接预测 raw flow |
| 单一时间尺度 | 抓不住短/中/长期事件影响差异 | Multi-scale temporal patching ×3 |
| 结果级 adapter 融合 | 13K 参数改不动 source 输出 | **特征级双分支 + 双向 cross-attention** |
| 单一全局 decay σ | 不同事件类型的衰减形状一致 | 三个并行 decay heads,decay 形状从事件特征学 |
| 不区分 affected/unaffected 解码 | §5.10 的发现没有反向 inform 架构 | Affected/Unaffected gate 路由 |

### 1.3 核心贡献(诚实 framing)

**重要**: 本架构每个单独组件在文献中都有先例 — 双流(STSGCN/STG-NCDE)、incident cross-attention(IGSTGNN)、multi-scale patching(MSD-Mixer)、learnable adjacency(ASTGCN)。

我们的 contribution 是 **"specific combination tailored for post-incident forecasting" + "ablation 量化每个组件对 affected-node MAE 的贡献"**,而非声称单个组件原创:

1. **特征级双流融合的 incident-aware 实例化**: 不同于 STSGCN 的两路在 spatial/temporal 分支,我们的 Normal/Incident 分支按"周期 vs 事件残差"功能分工,通过 sparse-masked bidirectional cross-attention 在节点维度互换信息
2. **多 decay head 替代 IGSTGNN 单一全局 σ**: 证明不同事件类型(7 类)需要不同 decay 形状
3. **学习 D 张量替代 IGSTGNN 预定义 D**: 证明从事件特征 + 路网信息中端到端学的关系张量能 outperform 手工特征
4. **Affected/unaffected gate 把 §5.10 反向 eval finding 反向用回架构**: 既是架构创新也是评估方法的呼应

每条都对应 §16.3 的一个主 ablation,贡献由实证支持。

### 1.4 与文献的关系

| 论文 | 借了什么 |
|---|---|
| IGSTGNN (KDD'26) | Cross-attention with incidents,Gaussian decay 概念 |
| STTIS (TKDE'23) | Region sampling 稀疏图思想(具体相似度量替换为路网距离) |
| MSD-Mixer (PVLDB'24) | Multi-scale temporal patching |
| Balance & Brighten (CIKM'25) | 双 student 训练范式(physics+data → 我们的 normal+incident) |
| FEDformer (ICML'22) | 频域感知作为 normal branch 的位置编码增强(可选) |
| XTraffic (NeurIPS'24) | 数据集本身,task formulation 合法性背书 |

---

## 2. 符号与维度

| 符号 | 含义 | 默认值 |
|---|---|---|
| `B` | batch size | 32 |
| `N` | 区域节点数 | Alameda 521 / CC 496 / Orange 990 |
| `T_h` | 历史长度 | 12 (=1 小时, 5 min/step) |
| `T_p` | 预测长度 | 12 (=1 小时) |
| `C_x` | 流量通道数 | 3 (flow, occupancy, speed) |
| `C_meta` | 传感器静态特征维度 | 26 (XTraffic) |
| `M` | 当前窗口内 active 事件数 | 上限 32(动态 padding+mask) |
| `C_e` | 事件特征维度 | **13** (one-hot 8 类型 + duration + severity + lat + lon + abs_postmile) |
| `d` | 模型 hidden dim | 64 |
| `K` | 时间尺度数 | 3 (短/中/长) |
| `L_enc` | 编码器层数 | 2 |
| `R` | 区域数 | 3 (Alameda, ContraCosta, Orange) |

---

## 3. 数据 Pipeline(必须重建)

### 3.1 为什么要重建

当前 H5 cache `outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5` 是**每事件 36 节点子图**结构,新架构需要**每 timestep 全县 N 节点**。

### 3.2 新 cache schema

每个样本 = (区域, sample_start_t),包含:

| 字段 | shape | 含义 |
|---|---|---|
| `region_code` | scalar | 0/1/2 = Alameda/CC/Orange |
| `sample_start` | scalar (= t) | 当前 timestep 索引;**预测窗口 = [t+1, t+12]**,与 IGSTGNN 锚点对齐 |
| `x_hist` | (N, T_h, C_x) | 历史 [t-T_h+1, t] 流量,N 是该区域全部节点 |
| `x_hist_mask` | (N, T_h, C_x) | **新增**: 历史输入的缺失 mask(1=有效, 0=插值填补) |
| `time_enc` | (T_h, d_t) | sin/cos(time-of-day) + sin/cos(day-of-week) + holiday flag,**对每个历史 step 独立** |
| `static_meta` | (N, C_meta) | 传感器 meta(per region 静态,可单独存复用) |
| `incident_feat` | (M_max, C_e) | active 事件特征,padded |
| `incident_mask` | (M_max,) | 1=真实事件, 0=padding |
| `y_true` | (N, T_p, C_x) | **horizons [t+1, t+12]** 的实际流量(对齐 IGSTGNN) |
| `y_baseline` | (N, T_p, C_x) | LearnedNormalRegion 在 [t+1, t+12] 的 pred(soft label,用于 L_normal) |
| `y_mask` | (N, T_p, C_x) | 有效位置 mask(传感器缺失置 0) |
| `affected_mask` | (N,) | 用 user 的 `node_labels.csv` 给的 affected partition。**仅 training-time 用作 loss masking 和 evaluation;NEVER 进 forward** |

**关键变更 vs v0.1**: 
- 锚点改为 `[t+1, t+12]` 与 IGSTGNN 一致 → head-to-head 直接对比 same horizons,论文不必解释错位
- 新增 `x_hist_mask` 处理 XTraffic 数据的缺失值

**Active 事件定义**: 起始时间在 `[t - 2 hours, t]` 范围内的事故 = active。**2 小时来自 XTraffic 论文 Fig 1 caption 的 "two-hour impact duration" 经验值**。每窗口最多 32 个(超出按"最近时间"top-32 截断)。

**实测(2026-05-12)**: Contra Costa 前 100 events 的 active 窗口内最多 8 个 events,M_max=32 充裕。

**事件类型**: 实测 XTraffic 2023 数据有 **8 个 type**(不是 7): `1141, Fire, NoInj, UnknInj, Hazard, AHazard, CarFire, Other`,因此 C_e=13 而非 12。

### 3.3 缺失数据处理

XTraffic 传感器存在非平凡缺失。处理:

```
对每个 (node, channel) 时间序列:
  1. 标记 NaN 位置 → x_hist_mask 中设 0
  2. 用 linear interpolation 填补内部 NaN
  3. 边界 NaN(首/末缺失)用 carry-forward / carry-backward
  4. 整段全 NaN(节点死)→ 填 0,且该节点的 y_mask 全 0(loss 不计)
```

**Forward pass 时**:
- `x_hist` 已是填补过的连续值
- `x_hist_mask` 作为额外通道拼接到输入: `x_hist_with_mask = concat([x_hist, x_hist_mask], dim=-1)`,实际输入通道数 = 2·C_x = 6
- 所有 Linear/MLP 的输入维度相应调整

### 3.4 数据规模估算

- Alameda: ~10 万 timestep × 521 nodes × 12 hist × 3 channel × 4 byte ≈ 7.5 GB
- 三区域合计 ~30 GB(可压缩 + lazy load)

### 3.5 实施脚本

| 脚本(待写) | 作用 | 复用现有 |
|---|---|---|
| `scripts/build_full_county_cache.py` | 主构建脚本 | 借 `build_test_raw_flow_cache.py` 框架 |
| `scripts/build_region_sampling_graph.py` | 预计算稀疏图 | 用 side cache 中的路网距离 |
| `scripts/run_normal_baseline_inference.py` | 出 LearnedNormalRegion 的 baseline | 借现有 LearnedNormalRegion 模型 |

---

## 4. Region Sampling Graph 构造

### 4.1 设计目标

- 提供稀疏邻接,降低 attention/GAT 复杂度
- 保证任意两节点 **≤ 3 跳可达**(信息全图可流;v0.2 修正,见 §4.3 实测)
- 反映**路网真实结构**(因事故影响沿路网传播)

### 4.2 算法(per region, 预计算一次)

**v0.3 重要变更**: shortcut 边从无向改为**有向** (hub→leaf only)。base k-NN 边仍是无向。理由见 §4.4。

```
Input: 路网距离矩阵 R ∈ R^(N×N) (从 archive/dis_matrix.npy)

Step 1: 物理 k-NN 基础层(双向)
  对每个节点 i,取 R[i, :] 中最近的 k=8 个 j 加边 (a, b) 双向
  → 物理邻居互相通信

Step 2: STTIS-style shortcut
  - core_count = ⌊√N⌋
  - 选 core: base k-NN 度数最高的 ⌊√N⌋ 个节点
  - core 之间全连接(双向)
  - 每个 leaf 连到路网距离最近的 3 个 core,**单向 hub → leaf**
    (hub 广播 regional context 给 leaf;hub 不被 leaf 灌输平均化)

Step 3: 组装为有向 edge_index
  base_pairs        → emit (a,b) AND (b,a)
  core_clique_pairs → emit (a,b) AND (b,a)
  hub_to_leaf       → emit (hub, leaf) ONLY

Output: edge_index ∈ Z^(2 × E_total),directed COO
        edge_index[0]=source (key/value), edge_index[1]=target (query)
```

### 4.3 实测结果(v0.3,2026-05-12)

```
=== Alameda  ===  N=521  edges=9010   in_max=45  out_max=408  ≤4-hop reach: 84%
=== CC       ===  N=496  edges=8899   in_max=43  out_max=377  ≤4-hop reach: 83%
=== Orange   ===  N=990  edges=17648  in_max=58  out_max=717  ≤4-hop reach: 77%
```

**与 v0.2(无向)对比**:
- max IN-degree: 717 → 58(**12× 降幅 ✓**)→ 直接根除 hub 的 attention softmax 稀释问题
- max OUT-degree: 不变 → hub 仍广播给 717 leaves,该有的覆盖度保留
- ≤3-hop pair 可达率: 100% → 50-58% (剩下经 4 hop 可达,信息流深度需要靠 multi-layer encoder)
- diameter: 3 → 大部分 4-5,少量更远

### 4.4 为什么有向化是正确的取舍

**Over-smoothing 风险来自 IN-degree**: 一个节点作为 target,要 softmax 注意力归一在它所有 source 上。717-way softmax 单个权重 ~0.0014,实际接近"对所有 source 求平均",节点表征退化为局部平均 → 多 hub 之间塌陷为相同。

**有向 hub→leaf 后**:
- Hub 的 IN-degree 只剩 ~45-58(物理 k-NN ~8 + 其他 hub ~22-31),attention 选择性恢复
- Hub 的表征由它附近的几个传感器 + 其他 hub 决定 → 真实反映该 hub 区域的状态
- Hub 仍向所有 leaves 广播(OUT-degree=717 保留)→ 区域上下文继续传播给远处节点

**Reachability 下降是可接受的**:
- 一个 leaf 是否需要直接知道远处另一个 leaf 的状态?**通常不**。
- Leaf 需要的是: (a) 自己的物理邻居 (k-NN 双向) + (b) 它的 region context (hub 广播)
- 跨区 leaf-to-leaf 的特定事件信息走 "leaf A → hub A → hub B → leaf B" 4 跳路径,通过 2 GAT 层 + 1 cross-attn = 3 hop 信息流**只能传一半**,后续 layer 补齐(我们将 L_enc=2 keep,看 PoC 决定要不要加 layer)

**信用归属**: 这个判断来自队友 review。我们感谢这条建议。

**性质**(理论 + 实测,2026-05-12):
- Degree: 均值 ~20, 中位数 17 ✓
- 任意两节点 **≤ 3 跳**可达(实测 100% reachable in ≤3 hops)
  - 注: 原设计目标 ≤ 2 跳是错的。实际只有约 65-70% pairs 是 2-hop reachable(共享 core 时);其余通过两个不同的 core 中转,3-hop reachable
  - **对架构无影响**: 2 层 GAT (in-branch) + 1 层 cross-attention = 3 跳信息传播,正好覆盖 diameter=3
- 边数 E ≈ N · √N + 8N ≈ O(N√N) ✓
- Degree 长尾: max 408-717,极高度数节点都是 cores(被很多 leaves 选为最近)
  - 真实路网中 freeway interchange 本来就是 hub,这是合理的

### 4.3 实测结果(2026-05-12 已生成)

```
=== Alameda  ===  N=521  edges=10120  diam≤3  max_deg=408  finite_dist=7.3%
=== CC       ===  N=496  edges=9970   diam≤3  max_deg=377  finite_dist=6.8%
=== Orange   ===  N=990  edges=19934  diam≤3  max_deg=717  finite_dist=4.2%
```

**输出文件**: `outputs/region_graphs/{region}_sparse_adj.npz`,字段含 `edge_index, region_idx, core_idx, degree_*, n_unreachable_2hop, n_unreachable_3hop` 等

---

## 5. Normal Branch

### 5.1 输入

```python
x_hist:    (B, N, T_h, C_x)
time_enc:  (B, T_h, d_t)         # d_t = 5
static:    (B, N, C_meta)
region_id: (B,)                   # 0/1/2
```

### 5.2 Multi-Scale Temporal Patching(long-focus weighted)

**v0.2 修正**: 
- scale_long 改为 patch=6/num=2,避免 num_patches=1 导致 mean pool 退化为 identity
- time_enc **不**再全局 mean — 改为每 patch 内分别 mean,broadcast 到对应 patch,保留 wall-clock 时序信息

```python
# 输入(已含 x_hist_mask)
# x_hist_with_mask: (B, N, T_h, 2*C_x)    # C_x=3 + mask channel = 6
# time_enc:         (B, T_h, d_t)

# 三个并行尺度
# scale_long:  patch_size=6, num_patches=2   → 抓全窗口左右半的 phase
# scale_mid:   patch_size=4, num_patches=3   → 20-min 块
# scale_short: patch_size=1, num_patches=12  → 原始 5-min step

embeds = []
for k in [long, mid, short]:
    P, P_size = num_patches_k, patch_size_k
    
    # 流量 patch
    x_p = x_hist_with_mask.reshape(B, N, P, P_size * 2*C_x)
    e_x = Linear(P_size * 2*C_x, d)(x_p)                      # (B, N, P, d)
    
    # 每 patch 内 time_enc 平均(关键修正)
    t_p = time_enc.reshape(B, P, P_size, d_t).mean(dim=2)      # (B, P, d_t)
    e_t = Linear(d_t, d)(t_p)                                   # (B, P, d)
    
    # 每 patch 加自己的 time encoding(broadcast 到节点)
    e_k = e_x + e_t.unsqueeze(1) + PosEnc(P, d)                # (B, N, P, d)
    
    # patch 维度 mean(scale_long 此时是 mean over 2 patches,不再退化)
    e_k_pooled = e_k.mean(dim=2)                                # (B, N, d)
    embeds.append(e_k_pooled)

# Long-focus mixing,init logits = log([0.5, 0.3, 0.2])
# (assert: softmax(init_logits) ≈ [0.5, 0.3, 0.2] in unit test)
α = softmax(self.scale_logits)
e_normal = sum(α[k] * embeds[k] for k in range(3))             # (B, N, d)

# 加 static + region embedding(time 已在 patch 内处理)
static_proj = Linear(C_meta, d)(static)                         # (B, N, d)
region_emb  = self.region_embed(region_id)                      # (B, d)
h_normal_in = LayerNorm(e_normal + static_proj + region_emb.unsqueeze(1))
```

### 5.3 Periodic Spatio-Temporal Encoder

```python
h = h_normal_in
for ℓ in range(L_enc):
    # 空间: 在 region-sampled graph 上做 GAT
    h_sp = GATConv(h, edge_index_sparse, heads=4, dim_per_head=d//4)
    h = LayerNorm(h + h_sp)
    
    # 节点级 FFN(channel mixing)
    h = LayerNorm(h + MLP(d → 4d → d)(h))

z_normal = h    # (B, N, d)
```

### 5.4 输出

```python
pred_normal_init = Linear(d, T_p * C_x)(z_normal).reshape(B, N, T_p, C_x)
```

(注: 最终 `pred_normal_final` 在 cross-attention 之后再算一遍,见 §8)

---

## 6. Incident Branch

### 6.1 输入

```python
x_hist:        (B, N, T_h, C_x)
incident:      (B, M_max, C_e)
incident_mask: (B, M_max)              # bool
static:        (B, N, C_meta)
rel_feat:      (M_max, N, 4)           # 预计算: [log_euclid, log_road, up/down, same_freeway]
                                         # per region 静态,batch 内广播
region_id:     (B,)
```

### 6.2 Learned D Tensor(事件-传感器关系张量)

```python
inc_emb     = Linear(C_e, d)(incident)                    # (B, M, d)
sensor_emb  = Linear(C_meta, d)(static)                   # (B, N, d)
rel_proj    = Linear(4, d)(rel_feat)                       # (M, N, d), broadcast to B

D_in = (
    inc_emb.unsqueeze(2).expand(B, M, N, d) +
    sensor_emb.unsqueeze(1).expand(B, M, N, d) +
    rel_proj.unsqueeze(0).expand(B, M, N, d)
)
D = MLP(d, 2d, d)(D_in)                                    # (B, M, N, d)

# 软 mask(用于 cross-attention 偏置)
attn_bias = Linear(d, 1)(D).squeeze(-1)                    # (B, M, N)
```

### 6.3 Multi-Scale Temporal Patching(short-focus weighted)

同 §5.2(含 patch-内 time_enc 修正 + scale_long patch=6/num=2),但 mixing 初值 `[0.2, 0.3, 0.5]`(短焦权重大)。输出 `e_incident: (B, N, d)`,**含独立的** region embedding(两分支的 region_embed 不共享参数)。

### 6.4 Incident-Aware ST Encoder

```python
h = e_incident
for ℓ in range(L_enc):
    # 空间 GAT(用同一张 region-sampled graph)
    h = LayerNorm(h + GATConv(h, edge_index_sparse, heads=4))
    
    # Incident → Sensor cross-attention
    Q = Linear(d, d)(h)                       # (B, N, d)
    K = Linear(d, d)(inc_emb)                 # (B, M, d)
    V = Linear(d, d)(inc_emb)                 # (B, M, d)
    
    scores = Q @ K.transpose(-1, -2) / sqrt(d)        # (B, N, M)
    scores = scores + attn_bias.transpose(-1, -2)     # 加学的 D 偏置
    scores = scores.masked_fill(~incident_mask.unsqueeze(1), -inf)
    attn   = softmax(scores, dim=-1)                  # (B, N, M)
    h_inc  = attn @ V                                 # (B, N, d)
    h      = LayerNorm(h + h_inc)

z_incident = h    # (B, N, d)
```

### 6.5 Three Parallel Decay Heads

**v0.2 修正**:
- M 维 sum 改为有界 `tanh(raw_sum)`,防多事件叠加无界放大
- τ/σ 单位明确为 **5-min step**(τ ∈ [1, 12] = 5 min … 1 hour)
- `scale_init_k` 默认: short=2 step (10 min), mid=6 step (30 min), long=12 step (1 hour);σ=12 时 envelope(τ=12) ≈ 0.6 是设计意图(长期影响不消亡)

```python
# 每个 head 学独立的 decay envelope shape + raw flow base
# decay envelope shared across channels (Q1 decision: per-channel base 已提供通道差异)
# τ, σ 单位: 5-min step

deltas = []
for k in [short, mid, long]:
    # 6.5a: 学每个 (incident, head) 的 decay 形状参数
    decay_params = MLP(C_e, 2)(incident)                       # (B, M, 2)
    σ_k, amp_k = decay_params.split(1, dim=-1)
    σ_k   = softplus(σ_k.squeeze(-1)) * scale_init_k           # step units; init: 2/6/12
    amp_k = sigmoid(amp_k.squeeze(-1))                          # ∈ [0, 1]
    
    # 6.5b: Gaussian 时间衰减包络(per-incident, per-time)
    τ = arange(1, T_p+1).float().to(device)                    # (T_p,) in step units
    envelope = amp_k.unsqueeze(-1) * exp(
        -τ.pow(2).unsqueeze(0).unsqueeze(0) / (2 * σ_k.unsqueeze(-1).pow(2))
    )                                                            # (B, M, T_p)
    
    # 6.5c: per-(incident, node) impact weight from D
    impact = sigmoid(Linear(d, 1)(D)).squeeze(-1)              # (B, M, N)
    
    # 6.5d: 聚合到 (B, N, T_p) — 用 tanh 有界 sum 防无界放大(v0.2 修正)
    raw_sum = einsum('bmn,bmt,bm->bnt', impact, envelope, incident_mask.float())
    node_decay = tanh(raw_sum)                                  # ∈ (-1, 1),保留叠加语义但有界
    
    # 6.5e: head 自己的 raw flow base
    pred_base = Linear(d, T_p * C_x)(z_incident).reshape(B, N, T_p, C_x)
    
    # decay 调制(envelope 跨 channel 共享,channel 差异由 pred_base 提供)
    delta_k = node_decay.unsqueeze(-1) * pred_base             # (B, N, T_p, C_x)
    deltas.append(delta_k)

delta_pred = sum(deltas)                                        # (B, N, T_p, C_x)
```

**为什么不用 max-pool over events**: 物理上多个上游事件可能加性堆队列(都向下游推拥堵),max 会丢叠加效应。`tanh(sum)` 保留加性语义,同时输出有界。

**为什么不用 softmax-over-M**: softmax 强制权重和为 1,丢失"事件越多影响越大"的绝对信号。

---

## 7. Bidirectional Sparse Cross-Attention

让 normal 和 incident 表征互相吸收信息,**用 region-sampled graph 的 edge_index 做真稀疏 attention**(v0.2 修正)。

**v0.1 错误**: 写成 `dense Q@K + masked_fill(-inf)`,这种写法 logits matmul 仍是 O(N²·d),mask 只省最终值不省计算。N=990 时单层 score matrix 占显存 ~500 MB(bfloat16),压力大。

**v0.2 正确实现**: 用 PyG 的 `MessagePassing` 框架,只对图中存在的边计算 attention logits:

```python
# 用 PyG 风格的 sparse cross-attention
# edge_index: (2, E),来自 region_sampled_graph;E ≈ N·√N

class SparseCrossAttn(MessagePassing):
    def forward(self, z_query, z_kv, edge_index):
        # z_query, z_kv: (B*N, d) — flatten batch into nodes,batch_idx 隐式
        Q = self.W_Q(z_query)
        K = self.W_K(z_kv)
        V = self.W_V(z_kv)
        # propagate 只在 edge_index 上算 attention
        out = self.propagate(edge_index, Q=Q, K=K, V=V)
        return out
    
    def message(self, Q_i, K_j, V_j, index, ptr, size_i):
        # Q_i: target node 的 query, K_j/V_j: source node 的 key/value
        score = (Q_i * K_j).sum(-1) / sqrt(d)        # (E,) — 真稀疏,无 N×N 矩阵
        alpha = softmax(score, index, ptr, size_i)   # PyG 的 segment softmax
        return alpha.unsqueeze(-1) * V_j

# 实例化
cross_attn_n = SparseCrossAttn(d)   # incident → normal
cross_attn_i = SparseCrossAttn(d)   # normal_updated → incident

# Step 1: normal queries from incident
z_normal_updated = LayerNorm(z_normal + cross_attn_n(z_normal, z_incident, edge_index))

# Step 2: incident queries from normal_updated
z_incident_updated = LayerNorm(z_incident + cross_attn_i(z_incident, z_normal_updated, edge_index))
```

**复杂度**(诚实): 
- 计算: O(E·d) per direction = O(N·√N·d) ≈ 30K · 64 = 2M ops/sample
- 显存(activation): O(E·d) ≈ 30K · 64 · 4 byte ≈ 8 MB per layer per sample,batch=32 即 ~250 MB,**可负担**

**只一层**(深了易 mode collapse,Q2 反复推敲后的结论)。

**实现注意**: 
- batch 内多样本要 concatenate node 维度成一个大图,用 PyG `Batch` 工具
- `edge_index` 要做 batch offset(每个样本的节点索引偏移 N)
- 实际写代码时用 PyG 的 `to_torch_csr` / `Batch.from_data_list` 处理

---

## 8. Affected/Unaffected Gate + Final Fusion

```python
# 8.1 Gate
g_in = concat([z_normal_updated, z_incident_updated], dim=-1)    # (B, N, 2d)
g_node    = sigmoid(MLP(2d, d, 1)(g_in)).squeeze(-1)             # (B, N)
g_horizon = sigmoid(MLP(2d, T_p)(g_in))                          # (B, N, T_p)
g = g_node.unsqueeze(-1) * g_horizon                              # (B, N, T_p)

# 8.2 用 update 后的 z_normal 重算 pred_normal(让 cross-attn 信息进入)
pred_normal_final = Linear(d, T_p * C_x)(z_normal_updated).reshape(B, N, T_p, C_x)

# 8.3 融合
pred = pred_normal_final + g.unsqueeze(-1) * delta_pred           # (B, N, T_p, C_x)
```

---

## 9. Loss Functions

### 9.1 三个 loss

**v0.2 修正**: L_normal 在 affected 节点 down-weight,因 y_baseline 是 LearnedNormalRegion 的 approximation,在 affected 节点上有噪声。

```python
# L_main: 全节点 raw flow MAE(主任务)
L_main = MAE(pred, y_true, mask=y_mask)

# L_normal: normal 分支对齐 LearnedNormalRegion baseline(soft label)
# 在 affected 节点 down-weight,在 unaffected 节点全权重
weight_per_node = where(affected_mask, λ_aff, 1.0)             # (N,), λ_aff=0.3
weight_full     = weight_per_node.unsqueeze(-1).unsqueeze(-1)   # (N, 1, 1) broadcast 到 (N, T_p, C_x)
L_normal = MAE(pred_normal_final, y_baseline, mask=y_mask, weight=weight_full)

# L_incident: 在 affected 节点上的 raw flow MAE(focus 事故)
incident_mask_full = y_mask & affected_mask.unsqueeze(-1).unsqueeze(-1)
L_incident = MAE(pred, y_true, mask=incident_mask_full)

# 总 loss
L_total = α * L_main + β * L_normal + γ * L_incident
```

### 9.2 默认权重

- `α = 1.0`
- `β = 0.3`(防 normal 分支被主 loss 拖去拟合 incident 残差)
- `γ = 0.5`(强化 affected 节点)
- `λ_aff = 0.3`(L_normal 在 affected 节点的相对权重,**v0.2 新增**)

**为什么不完全去掉 L_normal 在 affected 节点上的约束**(回应 review 1):
- L_main 是全节点的;若 affected 节点上 pred_normal 不被 L_normal 约束,L_main 会推 pred_normal 去拟合 y_true(含事件影响的真实流量)
- 结果: pred_normal 学到事件影响,**incident branch 失去存在意义**(反向 mode collapse)
- y_baseline 由 LearnedNormalRegion 出,本来就是训练来预测"无事件 counterfactual",其在 affected 节点上的 pred 恰恰是 pred_normal 该学的目标
- 折衷: down-weight 而非清零,承认 soft label 在 affected 上有噪声但保持监督信号

**可选改进**: Kendall 2018 uncertainty weighting,让 α/β/γ 可学。

---

## 10. Training Protocol

### 10.1 数据采样(per Q4 decision: 合训 + 区域 embedding)

- **区域分桶 batch**: 每个 batch 内只来自一个区域(N 一致,避免 padding 浪费)
- **采样策略**: 每个 epoch 内,按区域大小加权随机采样 batch(Orange 多采)
- **affected/non-affected 平衡**: 训练集中 affected 样本上采样 2× 防止 imbalance

### 10.2 Optimizer & Schedule

```
Optimizer:    AdamW(lr=1e-3, weight_decay=1e-5)
LR Schedule:  CosineAnnealingLR over total epochs, eta_min=1e-5
Batch size:   32
Epochs:       80, early stopping patience=10 on val L_main
Grad clip:    max_norm=1.0
Precision:    bfloat16 (mixed) on RTX 5080
```

### 10.3 Validation/Test Split

**复用现有 split**: train/val/test 与现有 H5 cache 一致(temporal split),避免 leakage。

---

## 11. 参数预算 + 显存估算

### 11.1 参数预算

| 模块 | 参数估算 |
|---|---|
| Multi-scale patching ×2 (normal + incident) | ~50K |
| GAT encoders ×2 (L_enc=2) | ~400K |
| Learned D tensor MLP | ~100K |
| Incident → sensor cross-attention | ~50K |
| 3 decay heads | ~150K |
| Bidirectional sparse cross-attention | ~30K |
| Gate | ~20K |
| Region embedding ×2 (normal + incident, 不共享) | ~400 |
| Final projection(pred_normal + pred_normal_final) | ~50K |
| **Total** | **~870K params** |

对比:
- 当前 Codex source `DualBranchSTTISGate`: ~5M params
- IGSTGNN: ~2-3M params
- **DIST-Net 比两者都轻**

### 11.2 激活显存估算(N=990, B=32, bfloat16)

最大的几个中间张量:

| 张量 | shape | 大小(bf16) |
|---|---|---|
| `D` (learned D tensor) | (B, M=32, N=990, d=64) | 130 MB |
| GAT 中间(L_enc=2 层 ×2 分支) | (B, N, d) per layer × ~5 stages | ~80 MB |
| Sparse cross-attention messages | (B, E≈30K, d) per direction | 8 MB × 2 = 16 MB |
| Decay heads pred_base ×3 | (B, N, T_p=12, C_x=3) ×3 | ~9 MB |
| Final pred 系列 | (B, N, T_p, C_x) ×3 | ~9 MB |
| Backward 梯度(2× forward 显存,粗估) | — | ~500 MB |
| **总激活显存(粗估)** | — | **~750 MB** |

**加上模型参数 + optimizer 状态(AdamW = 8x params)**:
- 870K params · 8 byte (state) ≈ 7 MB,可忽略

**显存总预算**: ~1 GB,RTX 5080 (16GB) 充裕。

**潜在显存炸点**: 
- 若 `D` tensor 改成 (B, M, N, 2d) 或更大 d,会线性放大
- 若 batch 加大到 64,激活几乎线性放大
- M=32 上限可能不够,若上调到 64,D tensor 翻倍

---

## 12. 实施路线图

**v0.2 修正**: 加 toy run + buffer,3 周改为 4-5 周(更现实估计)。

| 阶段 | 任务 | 预计耗时 | 输出 |
|---|---|---|---|
| **T+0** | 写 `build_region_sampling_graph.py` | 1 天 | `outputs/region_graphs/{region}_sparse_adj.npz` |
| **T+1~5** | 写 `build_full_county_cache.py`(含缺失数据插值) | 4-5 天(IO + 验证容易超时) | `outputs/dist_net/full_county_cache_{region}.h5` |
| **T+5~9** | 写模型代码: `dist_net/{normal_branch.py, incident_branch.py, cross_attn.py, gate.py, model.py}` | 4 天(含 PyG sparse attention) | 模块化 PyTorch 代码 |
| **T+9~10** | 写 `train_dist_net.py` + tensorboard logging(含 mode collapse 监控) | 1-2 天 | 训练脚本 |
| **T+10~11** | **Toy run**: ~50 节点 subset + 1 周数据,1 小时跑完 1 epoch,验证 forward/backward 数值正常、loss 下降、双分支 cosine 不接近 1 | 1 天 | 数值正确性确认 |
| **T+11~16** | PoC: Contra Costa 单区域训练 (496 节点最小) | 5 天(含 OOM/超参 retry buffer) | 单区域结果 + 选超参 |
| **T+16~25** | 合训三区域 | 7-9 天 | 合训模型 + checkpoint |
| **T+25~28** | Head-to-head 对比(复用 `compare_headtohead_igstgnn.py`)+ 主表 ablations | 3 天 | 对比表 + 论文 figure 数据 |

**总计**: **~4 周**(乐观)到 **5 周**(含 debug 实际更稳)到第一组 head-to-head 数字。

**Toy run 的价值**(reviewer 建议,接受):
- 提前发现 dimension/shape 错误(节省 1-2 周 debug)
- 验证 PyG sparse attention 是否真的省内存
- 验证 mode collapse 监控指标的 baseline 数值(双分支 init 时 cosine 是多少)

---

## 13. 复用现有 Stable Assets

| Asset | 复用方式 |
|---|---|
| `outputs/.../full_candidate_samples.h5` | **不直接复用**(36 节点不够),但 `y_residual` 公式参考 |
| `outputs/.../headtohead_igstgnn/test_raw_flow_side_cache.npz` | **复用**: `fut_scale_flow / normal_pred_flow` 当 L_normal 的 soft label 来源参考 |
| LearnedNormalRegion 模型 | **复用**: 生成新 cache 时跑 inference 出 `y_baseline` |
| `node_labels.csv`(affected/unaffected) | **复用**: L_incident 的 mask + 论文 §5.10 评估 |
| `compare_headtohead_igstgnn.py` | **复用**: 新模型出符合 schema 的 npz 即可 |
| 路网距离矩阵 | **复用**: 塞进 `rel_feat` + 用作 region graph 构造输入 |
| IGSTGNN test_predictions.npz | **复用**: head-to-head 直接对比 |

---

## 14. 设计决策记录(Decision Log)

四个关键决策的推敲过程,留档备查:

### Q1: Decay envelope 是否 per-channel?
- **决定**: 共享 envelope(per-incident-per-time,不分通道)
- **理由**: `pred_base_k` 已是 per-channel,通道差异由它表达;主指标 raw flow MAE 不依赖通道间精细对齐
- **Fallback**: 若 PoC 显示通道时间形状误差大,改为 `MLP(C_e → 2*C_x)` 学 per-channel 参数

### Q2: Cross-attention 形式?
- **决定**: Sparse-masked bidirectional cross-attention(用 region graph 当 mask)
- **理由**: 全 N×N attention 在 N=990 时 O(N²) 太贵;per-node pair 退化为 gated fusion 失去 attention 表达力;sparse-masked O(N√N) 可负担且保留 attention 语义
- **关键**: 只一层,深了 mode collapse

### Q3: Region sampling graph 怎么造?
- **决定**: 路网 k-NN 基础层 + STTIS-style core-leaf shortcut,**暂不引入**流量相似度
- **理由**: 节点级事故传播沿路网,不沿 profile 相似;路网距离已 precomputed
- **承认局限**(reviewer 提): 流量相似度也能反映 demand pattern 相近的远程节点(比如不同 OD 对的同时段高峰),完全丢可能损失这部分关联。**§16.3 可选 ablation 中评估**: "路网 only vs 路网+流量相似度 shortcut"
- **Fallback**: 若 core 选择导致某些节点 starvation,换 K-medoids 或图分区

### Q4: 多区域 — 合训 vs 分训?
- **决定**: 合训 + 区域 embedding(`nn.Embedding(R, d)`)
- **理由**: 论文叙事更强(generalizable architecture);数据量 3×
- **PoC 安全网**: 先 Contra Costa 单区域跑通,然后直接合训
- **Ablation**: 论文里给"合训 vs 分训"对比表

---

## 15. 已知风险与未决问题

### 15.1 已知风险

1. **数据 pipeline 重建工作量**: 30 GB 新 cache 是大工程,可能拖慢整体进度。**Mitigation**: 先小区域(CC)PoC 验证架构,再扩规模;toy run 阶段用 ~50 节点 subset 提前验证 pipeline 正确性。
2. **Mode collapse**(高优先级): 如果两分支退化为同一函数,双流就失去意义。**Mitigation**:
   - 输入差异: incident branch 看事件特征,normal branch 不看
   - 监督差异: L_normal 拉 normal 分支对齐 baseline,affected 节点 down-weight 但不清零(防反向 collapse)
   - 结构差异: short-focus vs long-focus patching weighting
   - 实时监控指标(见 §15.3),早期预警
3. **N=990 显存压力**: bfloat16 + batch 32 大概率够(见 §11.2 估算 ~1 GB),若 OOM 降到 batch 16。
4. **训练时间**: 一轮三区域 ~1 周,iteration 慢。**Mitigation**: toy run 阶段用 ~50 节点 + 1 周数据,1 小时一个 epoch 验证数值正确性。
5. **PyG sparse attention 实现复杂度**: PyG 的 MessagePassing API 与 batch 处理有非平凡 boilerplate。**Mitigation**: toy run 阶段必须验证 sparse attention 与 dense attention(在 toy 节点数下)数值等价。
6. **L_normal soft label 的局限**(reviewer 提出): y_baseline 在 affected 节点是 LearnedNormalRegion 的 approximation,有噪声。**Mitigation**: λ_aff=0.3 down-weight + 监控 pred_normal 在 affected 节点是否被推去拟合 y_true。

### 15.2 未决/可后续优化

- **频域增强**: FEDformer 风格的频域位置编码加到 normal branch 的 PosEnc 里,可能进一步强化周期先验。**优先级**: 中(若 PoC 显示 normal 分支欠拟合周期模式)
- **Per-channel decay**: Q1 的 fallback,看 PoC 结果决定
- **Uncertainty-weighted loss**: §9.2 的 α/β/γ 改为可学(Kendall 2018)
- **FiLM-style region modulation**: 若 PoC 显示 region embedding 192 params 不够,改为 region_emb 生成 scale + shift 调制每层 LayerNorm
- **Multi-task auxiliary heads**: 加 incident classification head,呼应 XTraffic 论文。**优先级**: 低
- **流量相似度补充 region graph**: Q3 决定不用,但若 ablation 显示纯路网图传播不足,可加流量相似度的远距 shortcut

### 15.3 训练监控指标(v0.2 新增)

每 N 步 log 以下指标到 tensorboard,作为 mode collapse 预警:

| 指标 | 健康范围 | 告警阈值 | 含义 |
|---|---|---|---|
| `cosine_sim(z_normal, z_incident).mean()` | 0.2 ~ 0.6 | > 0.85 | 接近 1 = 两分支表征塌陷为同一函数 |
| `g.mean()` | 0.1 ~ 0.5 | < 0.05 或 > 0.9 | 接近 0 = incident branch 失效;接近 1 = normal branch 被忽略 |
| `g[affected].mean() / g[unaffected].mean()` | > 2.0 | < 1.2 | gate 应在 affected 节点上明显大于 unaffected,否则 §5.10 finding 没被架构利用 |
| `MAE(pred_normal, y_baseline)[affected]` | 应低于 `MAE(pred, y_true)[affected]` | 反向 | 若 pred_normal 在 affected 上比 pred 还接近 y_true,说明 pred_normal 在学事件影响(反向 collapse) |
| `tanh saturation rate of node_decay` | < 30% | > 70% | tanh 饱和说明 raw_sum 过大,需检查 impact/envelope 是否过大 |
| Per-scale α (mixing weights) | 不漂移到边界 | 任一 α > 0.95 | 多尺度退化为单尺度 |

**告警时的 fallback 操作**:
- cosine 高 → 增大 L_normal 的 β 系数 (0.3 → 0.6);或在两分支间加更强结构差异
- gate 单边坍塌 → 检查 L_incident 的 γ 系数;检查 affected_mask 数据是否合理
- pred_normal collapse → 减小 affected 的 down-weight λ_aff (0.3 → 0.5)
- tanh 饱和 → 在 raw_sum 前加 LayerNorm 或除以 sqrt(M_active)

---

## 16. 评估方案

### 16.1 主指标(对标 IGSTGNN)

- Raw-flow MAE on **affected nodes only**(主指标)
- Raw-flow MAE on **all candidate nodes**
- Raw-flow MAE on **unaffected nodes**(副,看 normal 分支 work 不 work)

### 16.2 头对头对比

复用 `scripts/compare_headtohead_igstgnn.py`:
- 11 overlapping horizons(ours h=2..12 vs IGSTGNN h=1..11)
- 三个区域分别报(Orange 待传 IGSTGNN 预测)

### 16.3 Ablation 优先级(v0.2 重排)

#### 主表必跑(对应 §1.3 的四个 contribution claims):

| Ablation | 验证什么 contribution |
|---|---|
| 去掉 incident branch(纯 normal) | Contribution 1: 双流融合的价值 |
| 去掉 cross-attention(纯并联,直接 concat) | Contribution 1: 特征级融合 vs 简单并联 |
| 1 个 decay head vs 3 个 | Contribution 2: 多 decay head 的价值 |
| 用预定义 D(IGSTGNN 风格) vs learned D | Contribution 3: 学习关系张量 |
| 去掉 affected/unaffected gate | Contribution 4: §5.10 finding 反向 inform 架构 |
| **合训 vs 分训三个 region** | **验证 generalizability claim;若分训显著优于合训,论文 framing 要改** |

#### Appendix(放补充材料):

| Ablation | 验证什么 |
|---|---|
| 去掉 multi-scale patching(单尺度) | 多尺度的边际价值 |
| Region embedding vs FiLM modulation | region 调制方式选择 |
| L_normal 在 affected 节点 down-weight 系数 λ_aff sweep | 9.2 的折衷选择是否最优 |
| 频域增强(FEDformer-style PosEnc)on/off | 周期先验加强是否有用 |

#### 可选(写不下就跳过):

| Ablation | 验证什么 |
|---|---|
| Per-channel decay envelope | Q1 fallback 是否有改善 |
| Active 事件窗口 1h / 2h / 4h | 窗口长度 sensitivity |
| Region graph: 路网 only vs 路网+流量相似度 | Q3 决定的 sensitivity |

---

## 17. 与论文章节的映射

| 论文章节 | 对应文档 §  |
|---|---|
| §3 Method overview | §1.3 + §3 架构图 |
| §3.1 Region sampling | §4 |
| §3.2 Normal branch | §5 |
| §3.3 Incident branch | §6 |
| §3.4 Bidirectional fusion | §7 + §8 |
| §3.5 Training objective | §9 |
| §4 Experiments setup | §10 + §11 + §13 |
| §5 Results | §16.1 + §16.2 |
| §6 Ablations | §16.3 |
| §7 Related work | §1.4 |
