# 论文正文初稿：Latent-Incident Mediated Traffic Forecasting

## 摘要草稿

交通事故会显著改变道路交通状态，使得仅基于正常交通模式训练的预测模型在事故窗口中出现性能下降。现有事故感知交通预测研究通常依赖事故类型、事故检测或显式事故图建模，但 XTraffic 数据集中的实验表明，仅从交通状态中准确推断事故类型并不容易，且事故类型并不总能反映其真实交通影响。为此，本文提出一种潜在事故影响介导的时空交通预测框架，将未来交通状态分解为正常交通反事实预测与事故诱导残差两部分。模型首先通过 normal STGNN 学习正常交通状态，再在事故中心 full candidate sensor graph 上建模事故残差传播。进一步地，本文引入 learned-normal disagreement、dual historical residual 和 temporal decay head，以增强模型对事故影响强度和持续时间的刻画。实验结果表明，所提出模型在事故窗口中显著降低 robust MAE，最佳模型在 affected candidates 上将误差从 `1.2938` 降至 `1.1308`，提升 `12.60%`。分组分析进一步显示，temporal decay head 的收益主要集中在高严重度和长恢复时间事故中，说明显式建模事故影响持续性有助于提升事故场景下的交通预测性能。

## 1. Introduction 写作要点

交通流预测是智能交通系统中的核心任务，广泛服务于路径规划、拥堵预警和交通管理。已有时空图神经网络通过道路拓扑、传感器关联和历史交通序列学习正常交通规律，在常规场景中取得了较好效果。然而，道路事故会造成突发性的流量下降、速度下降、占有率上升，并引发上下游传播，使未来交通状态偏离正常交通模式。

一个直观思路是让模型学习事故类型或事故检测结果，但这一路线存在两个问题。第一，事故类型与交通影响并非一一对应：同类事故可能影响很弱，也可能造成长时间拥堵。第二，不同事故类型可能在交通状态上表现为相似的残差扰动。因此，对于交通预测任务而言，事故影响本身比事故类型更接近最终预测误差来源。

本文从这一观察出发，提出将事故影响建模为正常交通预测之外的时空残差。具体地，我们先学习正常交通反事实预测，再让事故分支专注学习事故诱导残差。与直接预测事故类别不同，该框架不要求模型在推理时准确识别事故类型，而是通过 normal branch disagreement、历史残差和事故上下文隐式刻画事故造成的交通扰动。

本文贡献可以概括为：

1. 提出 normal-impact decomposition，把事故场景下的交通预测分解为正常反事实预测和事故残差预测。
2. 在事故中心 full candidate sensor graph 上建模事故残差传播，避免依赖受影响节点标签进行 top-k 筛选。
3. 引入 learned-normal disagreement 和 dual historical residual，使事故分支能够更直接感知异常偏离。
4. 设计 temporal decay head，自适应建模事故影响随预测 horizon 的持续性。
5. 在 XTraffic 上验证该方法在 affected candidates、高严重度事故和长恢复事故中的有效性。

## 2. Method 正文草稿

### 2.1 Problem Formulation

给定历史交通状态序列 `X_{t-L+1:t}`、事故上下文 `c`、候选传感器图 `G=(V,E)`，目标是预测未来 `H` 个时间步的交通状态：

```text
Y_{t+1:t+H} = {flow, occupancy, speed}
```

不同于普通交通预测，本文关注事故窗口中的预测问题。事故发生后，观测到的未来状态可以被视为正常交通状态与事故扰动的叠加：

```text
Y = Y_normal + Delta_incident
```

因此模型预测形式为：

```text
Y_hat = Y_hat_normal + beta * G_decay * Delta_hat_incident
```

其中 `Y_hat_normal` 是正常交通反事实预测，`Delta_hat_incident` 是事故残差预测，`G_decay` 是 horizon-level temporal gate。

### 2.2 Normal Branch

Normal branch 使用正常或弱事故窗口训练，用于学习常规交通时空依赖。该分支不直接承担事故建模任务，而是提供事故窗口中的反事实参考。与统计 normal baseline 相比，learned normal STGNN 能更好捕捉区域内的时空模式，使残差目标更接近真实事故扰动。

在事故窗口中，normal branch 被用于生成：

```text
Y_hat_normal = f_normal(X, G)
```

随后事故分支学习：

```text
Delta_incident = Y_actual - Y_hat_normal
```

### 2.3 Incident Residual Branch

Incident branch 在 full candidate sensor graph 上运行。对于每个事故，候选节点由事故位置周围同区域传感器组成，而不是由受影响标签筛选。这一设计更符合实际推理场景，因为推理时无法提前知道哪些节点会受事故影响。

事故分支输入包括：

- 历史统计残差；
- learned-normal historical residual；
- `normal_delta`；
- `abs(normal_delta)`；
- 事故上下文；
- 候选节点距离、方向、postmile 和 anchor 标记；
- 候选节点图邻接关系。

其中 `normal_delta` 表示 learned normal prediction 与 statistical normal reference 之间的差异。其绝对值作为 disagreement proxy，用于提示 normal branch 对当前场景的不确定性或异常偏离。

### 2.4 Temporal Decay Head

事故影响通常具有持续性和恢复过程。短时轻微事故可能很快恢复，而严重事故可能在多个预测 horizon 上持续影响交通状态。为了刻画这种差异，本文在 residual STGNN hidden representation 上加入 temporal decay head：

```text
G_decay = 2 * sigmoid(phi(h))
```

其中 `h` 是节点级隐藏表示，`phi` 是小型 MLP，输出每个节点每个预测 horizon 的 gate。最终预测为：

```text
Y_hat = Y_hat_normal + beta * G_decay * Delta_hat_incident
```

该 gate 初始化接近 1，因此模型初始行为等价于普通残差融合；训练后 gate 可以学习不同事故、不同节点和不同预测步长下的影响持续性。

## 3. Experiments 正文草稿

### 3.1 Dataset and Setting

实验基于 XTraffic 数据集。数据包含多区域交通传感器时序、道路属性和事故记录。本文选取事故窗口构造 incident-centered candidate graph，并使用未来 12 个时间步作为预测目标。评价指标为 normalized residual space 上的 robust MAE，分别在 all candidates、affected candidates 和 unaffected candidates 上报告。

### 3.2 Baselines and Variants

实验比较以下模型变体：

1. statistical normal + residual STGNN；
2. learned normal residual model；
3. learned normal + `normal_delta`；
4. learned normal + dual historical residual；
5. learned normal + disagreement proxy；
6. learned normal + temporal decay head。

其中最后一个模型为本文当前最佳模型。

### 3.3 Main Results

主结果见 `main_result_table`。统计 normal residual STGNN 已经能够显著改善事故窗口预测，说明 residual impact modeling 是有效的。将 normal branch 替换为 learned normal 后，baseline robust MAE 本身下降，残差模型进一步改善预测。最佳模型在 all candidates 上将 robust MAE 从 `0.8328` 降至 `0.7239`，在 affected candidates 上从 `1.2938` 降至 `1.1308`。

这一结果表明，模型并非只改善未受影响节点上的普通预测误差，而是在真正受事故影响的候选节点上也能稳定降低误差。

### 3.4 Ablation Study

组件消融表明，`normal_delta` 和 dual historical residual 是主要增益来源。`normal_delta` 提供 learned normal 与 statistical normal 的差异信号，使事故分支更容易识别当前交通状态相对正常模式的偏离。Dual historical residual 则对齐了历史输入残差与未来预测残差，带来更明显的 affected candidates 改善。

`abs(normal_delta)` 作为 disagreement proxy 带来轻微额外收益。Temporal decay head 对 all candidates 的提升较小，但对 affected candidates 的改善更加明显，说明其主要作用集中在事故影响建模而不是普通背景预测。

### 3.5 Robustness across Seeds

在 seeds `7 / 11 / 23` 上，temporal decay model 的 all candidates robust MAE 为 `0.7242 +/- 0.0016`，affected candidates robust MAE 为 `1.1342 +/- 0.0041`。较小的标准差说明结果不是单一随机种子的偶然现象。

### 3.6 Group Analysis

为了理解 temporal decay head 的作用，本文进一步按照事故严重程度和恢复时间进行分组。结果显示，高严重度事故中 affected candidates 的 MAE 从 `1.3345` 降至 `1.3206`，长恢复事故中从 `1.2339` 降至 `1.2224`。相比之下，低严重度事故中 temporal decay head 略有退化。

这一现象符合模型设计动机：temporal decay head 主要帮助捕捉持续时间长、影响强的事故扰动；对于轻微事故，显式持续性建模可能引入额外噪声。因此，分组结果支持本文关于事故影响持续性建模的解释。

## 4. Discussion 草稿

本文结果说明，事故感知交通预测不一定需要把事故类型识别作为中心任务。更直接的方式是学习事故对交通状态造成的残差影响。Normal branch 提供反事实参考，incident branch 学习偏离正常模式的扰动，temporal decay head 则进一步刻画扰动随预测步长的持续性。

当前模型仍有改进空间。首先，low-severity group 中 temporal decay head 出现轻微退化，说明未来可以引入 confidence-aware gate，使模型在弱事故场景中自动降低事故残差强度。其次，当前 temporal decay 是从 hidden representation 中学习的隐式 gate，后续可以进一步加入 recovery-conditioned 或 survival-style head，以更显式地建模事故恢复过程。

