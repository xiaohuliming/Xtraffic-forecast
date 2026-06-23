# RGDN 去季节化残差引导双支网络 — 设计文档

日期 2026-06-17。状态:已与用户对齐设计,待写实现 plan。命名 RGDN 为暂定,可改。

## 1. 背景与动机

研究方向从事故标签导向收敛到分解导向,原因是事故标签在我们全部实验里零增益,详见
`项目总结报告_2026-06-10.md` 与记忆 `project_audit_experiments`。本设计是对现有
FourierDualNet 的一次结构性重构,不是再加模块。

它要修复的是一个已落盘的发现:旧双支线在等参数消融下,路由净效应只有约 0.7%,
原因是两条支线在做同一件事,对两个高度相关的 FFT 分量各做一遍图扩散,拆开几乎不带新信息。
数据见 `outputs/diagnostics/gwn_routing_vs_capacity.txt`。

RGDN 第一次给两条支线互不重叠的职责:主支线管节点自身的周期节律,残差支线管唯一真正在
路网里传播的东西即扰动。该设计与我们另一条证据线一致:STID 与 STAEformer 这类几乎不靠图
扩散的模型在 XTraffic 最强,说明周期基线本质是节点局部的,不需要跨节点图查询。全程
label-free,残差天然就是事故等异常扰动的代理。

## 2. 要检验的假设

1. 周期基线对每个节点是局部的,主支线对自身主流量做跨节点图扩散是浪费容量。
2. 在路网里真正空间传播的是扰动即残差,图结构应该用在残差支线。
3. 局部正常预测会随来袭扰动而变,这是局部乘扰动的交互效应,纯加法融合抓不到,需要把邻居
   残差摘要注入主支线。
4. 把已知周期基线当骨架直接加回,只预测偏离量,目标方差更小因而更好训更好讲。

## 3. 端到端数据流

```
历史流量 flow_hist (B,N,T_h)            未来 [day_kind,tod]
       │                                     │
       │ 查 baseline_median[day_kind,tod]     │ 查表(已在batch里=y_baseline)
       ▼                                     ▼
 baseline_hist=x_baseline ─减─ res_hist  baseline_future=y_baseline  已知骨架
                       │                              │
        ┌──────────────┴───────────────┐             │
   注入: 一步自适应邻接图卷         残差支线 GWN          │
     邻居残差摘要 (B,N,T_h,c)      gcn_bool=True         │
        │                              │             │
        ▼                              ▼             │
   主支线 GWN gcn_bool=False ─► Δ̂_local  Δ̂_spatial    │
        │                              │             │
        └─────────────相加─────────────┘             │
                       ▼                             │
             Δ̂ = Δ̂_local + Δ̂_spatial                 │
                       └───────────加回基线───────────┘
                                 ▼
                       ŷ = baseline_future + Δ̂
                  损失 = masked MAE(ŷ, flow_future)
```

基线固定不可学,梯度只流过 Δ̂,等价于让网络回归 `flow_future − baseline_future`。

## 4. 组件细节

### 4.1 气候态基线:直接复用缓存,不重算

读 `dist_net/data.py` 与 `scripts/build_full_county_cache.py` 后确认,管线已内置气候态基线,
直接复用即可,这一块零新建模代码且无泄漏。

- `RegionData.baseline_median` 形状 (2,288,N,3),按 [day_kind, tod] 索引,day_kind 为工作日
  与周末两档,tod 为 288 个 5 分钟槽。`baseline_scale` 同形状,为 MAD 稳健尺度。
- 缓存构建已保证两点:基线只用年度前 70% 即训练段拟合,`train_valid[train_cutoff:]=False`,
  无泄漏;且对事故窗口做了 masking,基线是去掉事故影响的正常模式,正合"正常状态流量"。空桶
  已回退为节点或全局中位数,尺度空桶回退为 1.0。
- 未来基线 `y_baseline` (N,T_p,3) 已由 get_sample 放进每个 batch,即"已知骨架"。
- 历史基线:给 get_sample 增补一个 x_baseline (N,T_h,3),按 [day_kind, tod] 同样查
  baseline_median 得到。仅新增一个 key,向后兼容,其他模型忽略即可。
- 残差标准化:`res = flow − baseline`,除以训练段残差的 masked 标量标准差 sd_res 得标准化残差;
  网络输出乘回 sd_res 得 Δ̂ 原始单位;`ŷ = y_baseline + Δ̂`。per-bin 的 baseline_scale 留作后续增强。
- 基线键是 (day_kind, tod) 两档日型而非完整 7 天,这是管线既有约定,沿用以复用验证过的基础设施,
  完整 dow 留作后续消融。

### 4.2 残差支线 ResidualBranch

- `gwnet(gcn_bool=True, addaptadj=True)`,复用 `baselines/GraphWaveNet/model.py`。
- 输入 in_dim=1,只喂标准化 flow 残差;occ 与 speed 作为后续消融,v1 不喂。
- 输出 Δ̂_spatial 形状 (B,N,T_p),为空间传播的扰动预测。

### 4.3 主支线 MainBranch

- `gwnet(gcn_bool=False)`,纯膨胀时序卷积,节点局部,不做跨节点扩散。
- 输入 in_dim = 1 加 2 加 c_inject:本节点标准化 flow 残差 1 通道,tod 与 dow 时间特征
  2 通道,注入的邻居残差摘要 c_inject 通道。
- 输出 Δ̂_local 形状 (B,N,T_p),为节点局部的扰动修正。
- 图信息只经注入这一条通道进入,骨干本身不含图扩散,这是与 4.2 的本质区别。

### 4.4 注入模块 InjectionGraphConv

- 自带小的自适应邻接:节点嵌入 E1,E2 形状 (N,d_emb),d_emb 默认 10,
  `adp = softmax(relu(E1 @ E2ᵀ), dim=1)` 形状 (N,N)。
- 对每个历史步 τ 做一步聚合 `summary[:, :, τ] = adp @ res_std[:, :, τ]`,得 (B,N,T_h)。
- 再用 1×1 conv 把 1 通道映射到 c_inject 通道,拼进主支线输入。
- 节点重要性即学到的 adp 权重。去掉本模块,主支线即纯局部,对应消融 V2。
- 参数约 `2·N·d_emb` 加一个极小 conv,N=521 时约一万参数,可忽略。

### 4.5 融合与损失

- 直接相加 `ŷ = baseline_future + Δ̂_local + Δ̂_spatial`。
- 不用 gated fusion,上次 D3 已证其无增益,去季节化结构下相加最自洽且不引入双计。门控
  留作可选消融,不在 v1。
- 损失 masked MAE 对 flow_future,沿用现有 all 与 affected 与 unaffected 三档。
- 产物 npz schema 与现有脚本完全一致:region_code, sample_start, region_node_idx,
  pred_raw_flow, actual_future_flow, y_mask_flow, affected_mask,以便直接接入
  `significance_tests.py` 与 `incident_type_breakdown.py`。

## 5. 参数对齐与决定性消融

铁律:所有变体总参数对齐单个 GWN 预算 P,跑前打印参数核对到个位百分比。这条来自基准那次
2P-vs-P 的教训,记忆 `project_audit_experiments` 与 CLAUDE.md 14.3 节。

| 变体 | 内容 | 隔离的问题 |
|---|---|---|
| V0a | 单 GWN,原始 flow | 经典基线 |
| V0b | 单 GWN,去季节化,基线加预测偏离 | 去季节化本身有没有用 |
| V1 RGDN | 基线 加 主支局部加注入 加 残差支图 | 完整机制 |
| V2 | V1 去掉注入,主支纯局部 | 注入有没有用 |
| V3 | V1 但主支线也开图卷,即旧对称双支 | 验证主支线开图卷是浪费 |
| V4 | V1 去掉去季节化,原始 flow 进两支 | 去季节化在双支结构里的贡献 |

参数对齐做法:先定 V0 单 GWN 标准配置并数出 P,再调两支线的 nhid 使两支加注入之和落在 P 的
几个百分点内。主支线 gcn_bool=False 每 nhid 更省参,故两支 nhid 不必相等,以参数对齐为准,
跑前打印记录。

读法:V1 要在参数对齐下显著超 V0b,超过种子噪声带约 0.04 到 0.08,才算机制成立。落在噪声
带内就是诚实负结果,照样写入论文,与项目一贯做法一致。

## 6. 评测协议与范围

- 第一轮:Alameda 单种子 42,跑 V0a 与 V0b 与 V1 与 V2,验证机制能否立住。每 run 在 5080 上
  预计小于 1 小时。
- 第二轮:机制立得住再补 V3 与 V4,补种子 42 与 1 与 2 出噪声带,再扩 ContraCosta 与 Orange,
  最后与 STAEformer 同台。
- 指标:masked MAE 的 all 与 affected 与 unaffected 三档。
- 显著性:复用 `significance_tests.py` 做按窗配对检验。
- 训练超参沿用现有 GWN 与 FDN 脚本:Adam,lr 1e-3,weight_decay 1e-4,cosine 调度,
  grad clip 5,epochs 与 GWN 基线一致以保证公平。

## 7. 实现落点

- 复用:`baselines/GraphWaveNet/model.py` 的 gwnet 带 gcn_bool 开关;`dist_net/data.py` 的
  MultiRegionDataset 与 make_loader;训练管线照 `scripts/train_gwn_spectral.py` 的多变体范式。
- 新建:`fourier_dual_net/rgdn.py`,含 ClimatologyBaseline 与 InjectionGraphConv 与 RGDN
  主模型;`scripts/train_rgdn.py`,带 `--variant v0a/v0b/v1/v2/v3/v4` 与 `--region`
  与 `--seed`。
- 远程执行按 CLAUDE.md 的 5080 操作手册:隐藏计划任务加 VBS 包装,创建后立即删除触发器。

## 8. 实现前必须核对的点

1. 基线复用已确认:管线用 (day_kind, tod) 键,baseline_median 已 train-only 即前 70% 且事故
   masked,无泄漏。未来基线用 batch["y_baseline"],历史基线给 get_sample 增补 x_baseline,
   均按 [day_kind, tod] 查 baseline_median。不再自算气候态。
2. 参数对齐:V0 单 GWN 的 P 数出来后,再定两支 nhid,跑前打印全部变体参数核对到个位百分比。
3. gwnet 在 gcn_bool=False 下退化为纯 TCN 的行为正确,nodevec 与 gconv 在该模式下不创建、
   走 residual_convs,确认无图参数。

## 9. 风险与诚实预期

- 可能落在种子噪声带内,与之前五次架构增强一样。去季节化加非对称结构是迄今最有原理的差异,
  但不保证有增益。
- 气候态基线 V0b 本身可能已很强,吃掉大部分可预测信号,压缩头部空间。无论结果如何都是有信息
  的:它直接量化了去季节化的价值。
- 注入可能与残差支线冗余,V2 专门检验这一点。
- 任何正向结论都以参数对齐为前提,严禁再出现容量混淆。

## 10. 后续上限实验,不在 v1

- 异构骨干第二阶段:主支线换 STID 式嵌入 MLP 或时序 Transformer,残差支线保留或换空间注意力。
- 给残差支线加 occ 与 speed 辅助通道。
- 可学习气候态基线,从预计算值初始化后微调。
- 给残差支线加去季节化辅助监督,鼓励局部与空间的分工。
- 气候态基线的日型轴细化:现在只有工作日与周末两档。可加节假日类,需要加州 2023 假日历,与或
  完整 7 天 dow。改 build_robust_baseline 的 day_kind 计算,把表第一维从 2 扩到 3 或更多,只需
  重算并回写 baseline_median 与 baseline_scale,flow 序列不变。注意稀疏:一年仅约 10 个节假日,
  单独建桶样本太少、易触发 min_count 回退,折中是把节假日并入周末桶,或仅做完整 dow,每类约 52
  样本统计更稳。何时做:全滑窗评测 2a 测试集含节假日时,或把去季节化升格为正式贡献时。当前对
  事件锚定测试集边际很小,先备着不做。
