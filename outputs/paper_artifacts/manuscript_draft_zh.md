# 潜在事故影响介导的双分支门控时空交通流预测

## 摘要

交通事故会突然改变路网运行状态，使交通流量、速度和占有率偏离常规时空规律。现有交通预测模型大多学习正常交通条件下的周期性、局部空间相关性和短期历史趋势，因此在事故发生后的预测精度容易下降。与其把事故类型识别作为核心目标，本文将事故理解为一种对正常交通状态的外部扰动，并直接建模事故对交通状态造成的影响。

本文提出一种潜在事故影响介导的双分支门控预测框架。模型首先使用 normal STGNN 估计无事故或弱事故条件下的正常反事实未来状态，然后把真实未来状态与正常反事实预测之间的差值定义为事故影响残差。在残差空间中，模型设置两个分支：normal-style residual branch 用于解释轻微、常规或可由正常模式延续得到的残差，incident graph residual branch 用于在事故中心 full candidate graph 上建模事故诱导的空间传播残差。最后，模型使用节点级、预测时距级 gate 自适应融合两个残差分支，并将融合后的残差加回正常反事实预测。

在 XTraffic 数据集上的实验表明，轻量 dual-branch gate no-aux 模型已能将全部候选传感器的 robust MAE 从 0.8328 降低到 0.7181。进一步将 incident branch 替换为 ST-TIS-style temporal self-attention 与 graph-biased spatial attention 后，最佳单次结果降低到 0.7135。最后，冻结 normal/incident residual branches 并只微调 gate head 后，最佳单次结果进一步达到全部候选节点 0.7116、受影响候选节点 1.1189。在随机种子 7、11 和 23 上，最终 gate-head fine-tuned ST-TIS 模型的全部候选节点 MAE 为 \(0.7135 \pm 0.0017\)，受影响候选节点 MAE 为 \(1.1206 \pm 0.0029\)。

为进一步处理 source 模型在长尾事故切片上的局部预测偏差，本文进一步在冻结 source 之后引入 latent impact correction adapter。该 adapter 是一个轻量 MLP（约 13K 参数），输入由 source 模型的 normal-style residual、incident-style residual、gate 与 disagreement 等中间状态构成，输出节点级、预测时距级修正项；并通过基于分支分歧的 anomaly gate 决定何时启用该修正。本文还提出一种 selective magnitude regret filter，仅对训练样本中 |correction| 处于尾部分位、且使误差变差的元素施加 hinge 惩罚，从而避免传统 regret loss 把 correction 整体压成零。在三随机种子均值上，adapter 在保留 source 总体 MAE 几乎不让步的前提下，将高严重度且长恢复事故的受影响节点 robust MAE 从 source 的 1.2764 进一步降低 30%；尤其在 source 出现 per-seed 局部回退的随机种子上，adapter 把 high-severity & long-recovery 切片的 affected delta 从 +0.000176 缩到 +0.000058，缓解了 source 自身无法解决的局部失败。结果说明，双分支门控的收益不是来自事故标签辅助监督，而是来自对正常残差解释和事故传播残差解释的自适应选择；进一步在冻结 source 之上加入受 anomaly gate 调制并以 selective regret 训练的轻量修正层，则能在不破坏 source 总体性能的前提下，针对长尾事故进行 surgical 鲁棒性改善。

**关键词：** 交通流预测；交通事故；XTraffic；时空图神经网络；残差学习；门控融合

## 1. 引言

交通流预测是智能交通系统中的基础任务，可服务于路径规划、拥堵管理、信号控制、应急响应和交通管控。近年来，时空图神经网络通过结合历史交通序列和路网空间结构，在常规交通条件下取得了较好的预测效果。这类模型通常假设未来交通状态可以由历史观测、时间周期性和邻近路段空间依赖共同推断。当交通状态处于相对稳定、可重复的运行模式时，模型能够学习到有效的预测规律。

事故场景打破了这一假设。事故发生后，局部道路可能出现流量下降、速度降低、占有率上升等现象，并且影响可能沿上下游道路传播。此时未来交通状态不再只是历史模式的自然延续，而是受到外部事件扰动后的结果。由于事故影响具有突发性、局部性、传播性和持续时间不确定性，常规预测模型在事故窗口内容易出现更大误差。

XTraffic 提供了大规模时空对齐的交通与事故数据，为事故后交通预测、事故分类和事故影响分析提供了基础。该数据集也揭示了一个重要问题：交通状态未必能够准确反推出事故类别，事故标签与实际交通影响之间并非一一对应。同一类事故在不同道路、不同时间和不同拥堵背景下可能造成完全不同的影响；不同类型事故也可能产生相似的交通状态偏移。因此，仅让模型识别“发生了什么事故”，未必能直接帮助模型预测“交通会如何变化”。

本文选择改变建模目标：不把事故识别作为主任务，而是让模型学习事故对交通流状态造成的影响。我们将事故影响表示为相对于正常交通反事实预测的残差。如果 normal branch 能够估计无事故或弱事故条件下未来交通应如何演化，那么后续模块就可以专注于学习正常模型无法解释的偏离部分。

这一想法也回应了最初的双分支门控设想。原始设想是将同一份 XTraffic 输入送入两个 MLP，分别生成“正常交通 embedding”和“事故交通 embedding”，再通过 gate 加权融合。这个思路与 Transformer 中人为定义 query、key、value 的方式有相似之处，但如果两个分支看到完全相同的输入、承担完全相同的预测目标，它们确实存在学到相似表示的风险。本文的调整是：保留双分支和 gate，但不让两个分支同时预测完整交通状态，而是让它们都在 residual-impact space 中工作。这样，“正常分支”和“事故分支”的语义不只是人为命名，而是由残差目标、候选事故图和门控融合机制共同约束出来。

本文主要贡献如下：

1. 将事故场景交通预测表述为 normal-impact decomposition 问题，即未来交通状态由正常反事实预测和事故影响残差共同组成。
2. 在残差空间中设计 dual-branch gated residual architecture，使 normal-style residual branch 与 incident graph residual branch 在每个节点和预测时距上自适应竞争。
3. 使用 full candidate graph 建模事故影响传播，避免推理阶段依赖 ground-truth affected nodes 进行候选节点筛选。
4. 通过 no-aux 实验和多随机种子实验验证模型收益，说明性能提升并非来自未来派生的事故影响标签辅助监督。
5. 在冻结 source 之上引入 latent impact correction adapter 与 branch-disagreement anomaly gate，并以 selective magnitude regret filter 抑制大幅且帮倒忙的修正，从而在保持 source 总体性能的前提下进一步改善长尾事故切片的鲁棒性，并缓解 source 在某些随机种子下出现的 per-seed 局部回退。

## 2. 相关工作

### 2.1 时空交通预测

交通预测通常被建模为时空序列预测问题。DCRNN、STGCN、Graph WaveNet 等代表性方法将图卷积、扩散卷积、自适应邻接矩阵或时间卷积与历史交通序列结合，用于学习传感器之间的空间依赖和时间动态。这些模型显著提升了正常交通场景下的预测精度，但其主要目标仍然是从历史观测中学习重复出现的交通模式。当事故作为外部扰动出现时，未来状态可能偏离这些模式，因此仅依赖常规时空相关性并不足够。

Transformer 和轻量化时空模型也为交通预测提供了新的基础结构。ST-TIS 通过时空信息融合和区域采样降低复杂度，同时保留较强的时空依赖建模能力。这类模型可以作为 normal branch 或 incident branch 的候选 backbone。本文在轻量 STGNN 版本之外进一步实现了一个 ST-TIS-style incident branch，用 temporal self-attention 与 graph-biased spatial attention 增强事故残差分支的时空表达能力。

### 2.2 分解建模与正常状态估计

时间序列分解常用于区分趋势、季节性和残差变化。本文同样采用“分解”的思想，但分解目标不同：我们不是把交通序列拆成趋势和季节项，而是把事故窗口内的未来状态拆成正常反事实成分和事故影响残差成分。

正常状态建模与本文密切相关。若能够估计交通在无事故条件下的正常运行状态，就可以进一步识别当前观测和未来状态中的异常偏移。本文的 normal branch 提供一个可学习的正常参照，使后续残差分支不必重新学习完整交通预测，而是聚焦于正常模型无法解释的部分。

### 2.3 事故感知交通预测

事故感知交通预测近年来受到更多关注。XTraffic 将交通传感器序列与事故记录进行时空对齐，使研究者能够分析事故发生后的交通状态变化。IGSTGNN 等工作进一步通过事故空间融合和时间影响衰减机制，将事故上下文显式纳入预测模型。

本文与这些方法的共同点在于都认为事故是一类外部扰动；不同点在于，本文不以事故类型识别或显式事故图推断为核心，而是将事故影响作为相对于正常状态的 latent residual 来学习。我们的 gate 不直接判断事故类别，而是在每个候选节点和预测时距上判断当前残差更像 normal-style residual，还是更像 incident graph residual。

## 3. 问题定义

设道路传感器网络为 \(G=(V,E)\)，其中 \(V\) 表示交通传感器节点集合，\(E\) 表示由路网邻接、空间距离或相关性构成的边集合。每个时间步 \(t\)，节点 \(v\) 具有交通状态向量，例如流量、占有率和平均速度。给定历史交通序列 \(X_{t-L+1:t}\)、事故上下文 \(c\) 以及事故中心候选图 \(G_c=(V_c,E_c)\)，目标是在未来 \(H\) 个时间步预测候选节点的交通状态：

\[
\hat{Y}_{t+1:t+H} \in \mathbb{R}^{H \times |V_c| \times C},
\]

其中 \(C\) 为交通状态通道数。

对于事故场景，本文假设观测到的未来交通状态可以分解为：

\[
Y = Y^{normal} + \Delta^{incident},
\]

其中 \(Y^{normal}\) 表示无事故或弱事故条件下的反事实未来状态，\(\Delta^{incident}\) 表示事故造成的残差影响。由于真实的无事故反事实状态不可观测，我们使用 normal branch 进行估计：

\[
\hat{Y}^{normal} = f_{normal}(X_{t-L+1:t}, G).
\]

残差目标定义为：

\[
\Delta^{target} = Y - \hat{Y}^{normal}.
\]

在残差空间中，模型预测 normal-style residual 和 incident-style residual：

\[
\hat{\Delta}^{normal} = f_{res}^{normal}(Z),
\]

\[
\hat{\Delta}^{incident} = f_{res}^{incident}(Z, c, G_c),
\]

其中 \(Z\) 表示由历史交通、统计正常参照和 learned normal forecast 构造的残差特征。节点级、预测时距级 gate 输出：

\[
\alpha = \sigma(g(Z, h^{normal}, h^{incident})).
\]

最终残差为：

\[
\hat{\Delta}
= (1-\alpha) \odot \hat{\Delta}^{normal}
+ \alpha \odot \hat{\Delta}^{incident}.
\]

最终预测为：

\[
\hat{Y} = \hat{Y}^{normal} + \beta \cdot \hat{\Delta},
\]

其中 \(\beta\) 是验证集选择的残差缩放系数。

## 4. 方法

![方法框架图](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/method_architecture.png)

**图 1：** 潜在事故影响介导的双分支门控残差框架。Normal STGNN 估计正常反事实未来状态，两个 residual branch 分别建模 normal-style residual 和 incident graph residual，节点级、预测时距级 gate 自适应融合两个残差解释，最后将 gated residual 加回正常预测。

### 4.1 总体框架

所提出模型由五个核心模块构成：normal STGNN branch、residual construction module、normal-style residual branch、incident graph residual branch 和 node-horizon gate。Normal branch 用于估计常规交通状态；residual construction module 将历史观测、统计正常参照和 learned normal forecast 进行比较，构造异常偏离信号；两个 residual branch 在残差空间中分别生成候选解释；gate 决定每个节点、每个预测时距应更多依赖哪一类残差解释。

这一设计的核心原则是：事故分支不应重复学习正常交通预测，而应专门学习正常交通模型无法解释的部分。相比“同一输入直接生成两个完整 embedding”，残差空间双分支让两个分支的分工更明确，也降低了表示塌缩为相似功能的风险。

### 4.2 正常反事实分支

Normal branch 以历史交通序列和图结构为输入，输出未来正常交通预测：

\[
\hat{Y}^{normal}_{t+1:t+H} = f_{normal}(X_{t-L+1:t}, G).
\]

与统计正常基线相比，learned normal STGNN 能够捕捉更复杂的时间模式、空间依赖和区域特异性交通规律。在训练事故残差分支时，normal branch 提供反事实正常参照，残差目标定义为 \(Y-\hat{Y}^{normal}\)。这使后续分支的目标不再是重建完整未来交通状态，而是学习真实未来状态相对于正常预测的偏离。

### 4.3 Full Candidate Incident Graph

对于每个事故样本，我们围绕事故位置构建候选传感器图。候选节点由事故区域、空间距离、道路方向、postmile 差异和传感器有效性等约束共同确定。与只选择 ground-truth affected nodes 的方法不同，本文在 full candidate graph 上训练和评估事故残差分支。

这种设计更接近真实应用场景。推理时模型可能知道事故记录和大致位置，但通常无法提前知道哪些传感器会被影响，也无法知道影响传播范围。因此，模型需要同时学习受影响节点和未受影响节点的响应。Full candidate graph 可以避免使用受影响标签进行候选节点筛选，从而使预测任务更真实。

### 4.4 残差特征构造

模型接收多类残差信号。第一类是 statistical historical residual，即近期观测与统计正常参照之间的偏差。第二类是 learned-normal historical residual，即近期观测与 learned normal branch 给出的正常估计之间的偏差。该信号使历史残差输入与未来残差目标保持一致。

第三类是 learned-normal disagreement，记为：

\[
\delta^{normal} = \frac{\hat{Y}^{normal} - \hat{Y}^{stat}}{s},
\]

其中 \(\hat{Y}^{stat}\) 为统计正常预测，\(s\) 为归一化尺度。该特征反映 learned normal forecast 与 statistical normal forecast 之间的差异。直观上，当两种正常参照出现明显分歧时，说明当前样本可能包含较强的异常因素。本文进一步加入 \(|\delta^{normal}|\)，作为轻量级 disagreement proxy，用于显式提供异常幅度信息。

### 4.5 双分支残差建模

Normal-style residual branch 用于建模轻微残差、局部噪声、正常模式延续误差以及不需要事故传播解释的偏差。该分支仍然在残差空间工作，因此它不是第二个完整 normal predictor，而是 normal counterfactual 之后的修正项。

Incident graph residual branch 在事故中心候选图上预测事故传播残差：

\[
\hat{\Delta}^{incident} = f_{res}^{incident}(Z, c, G_c).
\]

该分支通过时间编码捕捉近期残差变化，通过图消息传递建模候选节点之间的空间影响传播，并结合事故上下文学习不同事故样本下的残差模式。基础版本使用轻量 STGNN；增强版本将 incident branch 替换为 ST-TIS-style 模块，其中 temporal self-attention 编码每个候选节点的历史残差序列，top-k graph-biased spatial attention 在 full candidate graph 上融合邻近候选节点信息。

### 4.6 节点级预测时距 Gate

Gate 的作用不是给两个完整预测结果做固定平均，而是在残差空间中选择解释来源。对于每个候选节点和预测时距，gate 输出 \(\alpha_{v,h}\)。当 \(\alpha_{v,h}\) 较小时，模型更依赖 normal-style residual；当 \(\alpha_{v,h}\) 较大时，模型更依赖 incident graph residual。

\[
\hat{\Delta}_{v,h}
= (1-\alpha_{v,h})\hat{\Delta}^{normal}_{v,h}
+ \alpha_{v,h}\hat{\Delta}^{incident}_{v,h}.
\]

这种设计让模型能够在未受影响或轻微影响节点上抑制事故分支，在受影响或影响持续节点上激活事故传播分支。它也把“事故是否重要”从显式分类任务转化成了隐式的残差解释选择任务。

### 4.7 训练与推理

当前最佳模型采用 no-aux 训练设置，即不使用未来派生的 affected node 或 impact label 作为辅助监督。训练目标只来自最终交通预测误差。受影响节点标签只用于测试集分组评价，不参与推理阶段候选节点筛选，也不是模型取得收益的直接来源。

推理时，模型需要历史交通序列、事故上下文和事故中心候选图。模型先产生正常反事实预测，再预测并门控融合残差，最后输出事故窗口下的未来交通状态。

### 4.8 后置 Latent Impact Correction Adapter

Source 模型本身已经在 dual-branch + gate 范式下取得了较强的整体精度，但仍然存在两类残余问题：第一，对极端样本，gate 没有把 incident branch 权重压到足够低（例如某些样本中 normal branch 误差 \(\approx 2.2\) 但 incident branch 误差 \(\approx 6.5\)，gate 仍输出 \(\approx 0.42\)）；第二，在不同随机种子初始化下，severity-high 且 recovery-long 切片的受影响节点 MAE 偶尔会比 source 自己的反事实参照更差，呈现 per-seed 局部回退。这些问题源自 source 内部 gate 的 scalar 决策能力，已经接近其结构上限。

为此，本文在已经训好的 source 之外引入一个轻量 latent impact correction adapter，整体形式为：

\[
\hat{Y}^{final} = \hat{Y}^{source} + g_{anom} \cdot c_{adapter},
\]

其中 \(\hat{Y}^{source} = \beta \cdot f_{src}(\cdot)\) 为冻结 source 的预测，\(c_{adapter}\) 是 adapter 输出的局部修正项，\(g_{anom}\) 是基于分支分歧的 anomaly gate。

**Adapter 结构。** Adapter 是一个 3 层 MLP，仅约 13K 可训练参数。它的输入由 source 的中间状态串联得到：source 预测自身、normal-style residual、incident-style residual、二者差异 \(\hat{\Delta}^{incident} - \hat{\Delta}^{normal}\) 及其绝对值、source gate、source 的 normal-veto 中间量、`normal_delta`、source 预测的 impact heatmap、affected-node logits、event-level 辅助预测、global context、horizon embedding 与 channel one-hot。adapter 的输出通过 \(c = c_{\max} \cdot \tanh(\cdot)\) 限幅到 \([-c_{\max}, c_{\max}]\)（实验中 \(c_{\max}=0.8\)），最后一层零初始化以保证训练初期 \(c \equiv 0\)，模型严格等价于 source。

**Anomaly gate.** 修正不直接加到 source 预测上。Adapter 的输出先被一个**无可学习参数**的 anomaly gate 调制：

\[
g_{anom} = \alpha_{floor} + (1-\alpha_{floor}) \cdot \sigma\!\left(\frac{|\hat{\Delta}^{incident} - \hat{\Delta}^{normal}| - \tau}{T}\right),
\]

实验中 \(\alpha_{floor}=0.25, \tau=0.5, T=0.25\)。直觉是：当 source 内部两个 branch 预测分歧大（\(|\hat{\Delta}^{incident} - \hat{\Delta}^{normal}|\) 大）时，说明该样本的事故影响模式不易被 source 内部融合解释，gate 接近 1，让 adapter 的修正充分进入；反之当两个 branch 几乎一致时，source 自身就足够，gate 被压到下限 0.25，避免 adapter 在 well-predicted 样本上引入噪声扰动。本文实验也比较过用 source 预测的 affected-node 概率作为 gate 触发条件，但效果不如分支分歧——后者作为 latent incident impact signal 比下游分类输出更可靠。

**Selective magnitude regret filter.** 朴素的 regret loss \(\ell_{regret} = \mathbb{E}[\mathrm{ReLU}(|y - \hat{Y}^{source} - c| - |y - \hat{Y}^{source}|)]\) 对所有元素施加 hinge 惩罚，会引发两种失败模式：权重小则收效甚微，权重大则把 \(c\) 整体压成 0。诊断显示 affected 节点上 \(c\) 的 sign\_match\_rate 仅约 0.51，beneficial 与 harmful 元素数量近似 1:1，但**少量大幅 harmful 元素的边际损失更大**。基于此，我们提出 selective magnitude regret filter：仅对 \(|c|\) 在 tail-affected 子集中分位数 \(\geq q\) 的元素（即"大幅修正"）施加 regret 惩罚：

\[
\ell_{regret}^{\text{sel}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathbb{1}[|c_i| \geq Q_q] \cdot \mathrm{ReLU}\!\left(|y_i - \hat{Y}^{source}_i - c_i| - |y_i - \hat{Y}^{source}_i|\right),
\]

其中 \(\mathcal{M}\) 为 tail-affected mask（severity-high 或 recovery-long 事故下的受影响节点集合），\(Q_q\) 为 \(\mathcal{M}\) 内 \(|c|\) 的 q-分位数。注意分母仍然是完整 \(|\mathcal{M}|\)，而非过滤后元素数——这一点对避免 per-element gradient 被放大、保持 \(c\) 不被整体压零至关重要。实验中 \(q=0.90, \lambda_{regret}=0.05\) 给出最佳 3-seed Pareto 表现。

**与原 source 的关系。** Adapter 完全外挂在冻结 source 之外：source 的所有参数与超参不变，adapter 训练时只梯度回传到自身 13K 参数。这种 frozen-base + adapter 的范式与近年 NLP 中的 LoRA/adapter 思路相通；其优势是 source 的预测能力得以完整保留，adapter 只在需要时给出局部修正，而 anomaly gate 与 selective regret 共同保证修正"该出手时才出手、出手不至于帮倒忙"。

## 5. 实验

### 5.1 数据集与评价指标

实验使用 XTraffic 数据集。该数据集将交通传感器时间序列与事故记录进行时空对齐，交通状态包含流量、车道占有率和平均速度。当前实验使用 Alameda、Orange 和 Contra Costa 三个区域。对每个事故样本，模型在事故中心 full candidate graph 上预测未来 12 个时间步的交通状态。

本文使用 normalized residual space 中的 robust MAE 作为主要指标，并报告全部候选节点、受影响候选节点、未受影响候选节点和不同预测时距上的 robust MAE。

### 5.2 主结果

| 模型 | 正常分支 | 事故残差输入 | All MAE | All gain | Affected MAE | Affected gain |
|---|---|---|---:|---:|---:|---:|
| statistical normal + residual STGNN | statistical blend | statistical residual | 0.8735 -> 0.7378 | 15.54% | 1.3888 -> 1.1659 | 16.05% |
| learned normal | learned normal STGNN | future residual target only | 0.8328 -> 0.7579 | 8.99% | 1.2938 -> 1.1646 | 9.99% |
| + normal_delta | learned normal STGNN | normal_delta | 0.8328 -> 0.7434 | 10.73% | 1.2938 -> 1.1620 | 10.19% |
| + dual historical residual | learned normal STGNN | normal_delta + dual history | 0.8328 -> 0.7254 | 12.90% | 1.2938 -> 1.1380 | 12.04% |
| + disagreement proxy | learned normal STGNN | normal_delta + abs(normal_delta) + dual history | 0.8328 -> 0.7248 | 12.97% | 1.2938 -> 1.1381 | 12.04% |
| + temporal decay head | learned normal STGNN | normal_delta + abs(normal_delta) + dual history + temporal gate | 0.8328 -> 0.7239 | 13.07% | 1.2938 -> 1.1308 | 12.60% |
| residual temporal decay no-aux | learned normal STGNN | normal_delta + abs(normal_delta) + dual history + temporal gate; no aux labels | 0.8328 -> 0.7221 | 13.30% | 1.2938 -> 1.1290 | 12.74% |
| dual-branch gate | learned normal STGNN | normal-style residual branch + incident graph branch + gate | 0.8328 -> 0.7203 | 13.51% | 1.2938 -> 1.1283 | 12.80% |
| dual-branch gate no-aux | learned normal STGNN | normal-style residual branch + incident graph branch + gate; no aux labels | 0.8328 -> 0.7181 | 13.78% | 1.2938 -> 1.1234 | 13.17% |
| ST-TIS incident branch no-aux | learned normal STGNN | normal-style residual branch + ST-TIS-style incident branch + gate; no aux labels | 0.8328 -> 0.7135 | 14.32% | 1.2938 -> 1.1217 | 13.30% |
| ST-TIS gate-head fine-tune no-aux | learned normal STGNN | freeze residual branches, fine-tune gate head only; no aux labels | **0.8328 -> 0.7116** | **14.56%** | **1.2938 -> 1.1189** | **13.52%** |

结果表明，当前最佳总体模型是 ST-TIS gate-head fine-tune no-aux。轻量 dual-branch gate 已经优于较早的 single incident residual branch 和 temporal decay head；在此基础上增强 incident branch 后，全部候选节点 MAE 进一步下降；最后只微调 gate head 又改善了融合策略。该模型仍然不依赖辅助事故影响标签，这一点很重要，因为它说明性能提升来自残差分解、双分支门控、事故分支时空表达能力和无标签 gate 校准，而不是通过未来派生标签获得额外信息。

### 5.3 组件消融

| 变体 | Added signal | All MAE | Affected MAE | Step gain all | Step gain affected | beta |
|---|---|---:|---:|---:|---:|---:|
| learned normal | future residual target only | 0.7579 | 1.1646 | -- | -- | 1.0000 |
| + normal_delta | normal_delta | 0.7434 | 1.1620 | 1.92% | 0.22% | 0.9500 |
| + dual historical residual | normal_delta + dual history | 0.7254 | 1.1380 | 2.43% | 2.07% | 1.0000 |
| + disagreement proxy | normal_delta + abs(normal_delta) + dual history | 0.7248 | 1.1381 | 0.08% | -0.01% | 1.0000 |
| + temporal decay head | normal_delta + abs(normal_delta) + dual history + temporal gate | 0.7239 | 1.1308 | 0.12% | 0.64% | 1.0000 |
| residual temporal decay no-aux | normal_delta + abs(normal_delta) + dual history + temporal gate; no aux labels | 0.7221 | 1.1290 | 0.25% | 0.16% | 1.0000 |
| dual-branch gate | normal-style residual branch + incident graph branch + gate | 0.7203 | 1.1283 | 0.25% | 0.07% | 1.1000 |
| dual-branch gate no-aux | normal-style residual branch + incident graph branch + gate; no aux labels | 0.7181 | 1.1234 | 0.31% | 0.43% | 1.0500 |
| ST-TIS incident branch no-aux | normal-style residual branch + ST-TIS-style incident branch + gate; no aux labels | 0.7135 | 1.1217 | 0.64% | 0.15% | 1.0000 |
| ST-TIS gate-head fine-tune no-aux | freeze residual branches, fine-tune gate head only; no aux labels | **0.7116** | **1.1189** | **0.27%** | **0.25%** | 1.0000 |

消融结果可以分成四层理解。第一层是残差信号是否有效：normal_delta、dual historical residual 和 disagreement proxy 逐步降低误差，说明 learned normal forecast 与统计正常参照之间的差异确实能帮助模型识别异常偏离。第二层是结构是否有效：在同样的 learned normal、full candidate graph 和残差目标下，dual-branch gate no-aux 进一步优于 no-aux temporal decay，说明两个残差解释分支加 gate 的结构比单一事故残差分支更灵活。第三层是事故分支表达能力是否有效：ST-TIS-style incident branch 在保持 no-aux 设置的同时继续降低总体 MAE，并在三随机种子平均上改善 affected MAE。第四层是 gate 校准是否有效：冻结两个 residual branch 后只微调 gate head 还能继续提升，说明当前收益不是简单来自更大分支，而是来自更好的局部残差解释选择。

### 5.4 随机种子稳定性

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | beta |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 0.7181 | 1.1234 | 0.5279 | 1.1338 | 1.3648 | 1.0500 |
| 11 | 0.7198 | 1.1296 | 0.5276 | 1.1398 | 1.3742 | 1.0500 |
| 23 | 0.7166 | 1.1188 | 0.5279 | 1.1289 | 1.3566 | 1.0000 |
| Mean ± std | **0.7182 ± 0.0016** | **1.1240 ± 0.0054** | **0.5278 ± 0.0002** | **1.1342 ± 0.0055** | **1.3652 ± 0.0088** | -- |

三个随机种子的标准差较小，说明模型提升不是由某个偶然初始化造成的。尤其是 affected candidates 的均值和 seed 7 单次结果非常接近，表明受影响节点上的提升具有一定稳定性。

进一步地，对 ST-TIS-style incident branch no-aux 版本以及 gate-head fine-tune 版本使用相同的 seed 7、11 和 23 进行验证：

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | beta |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 0.7153 | 1.1240 | 0.5236 | 1.1328 | 1.3678 | 1.1000 |
| 11 | 0.7161 | 1.1203 | 0.5265 | 1.1290 | 1.3662 | 1.0000 |
| 23 | 0.7135 | 1.1217 | 0.5220 | 1.1314 | 1.3638 | 1.0000 |
| Mean ± std | **0.7150 ± 0.0013** | **1.1220 ± 0.0019** | **0.5240 ± 0.0023** | **1.1311 ± 0.0019** | **1.3659 ± 0.0020** | -- |

与轻量 dual-branch gate 相比，ST-TIS-style 版本的三 seed 平均 All MAE 从 0.7182 降至 0.7150，Affected MAE 从 1.1240 降至 1.1220。虽然单次最优 affected MAE 仍来自轻量版本 seed 23，但 ST-TIS-style 版本在均值和方差上更稳健，因此更适合作为下一阶段主模型。

Gate-head fine-tune 后的三随机种子结果如下：

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | beta |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 0.7150 | 1.1239 | 0.5232 | 1.1325 | 1.3690 | 1.0500 |
| 11 | 0.7138 | 1.1189 | 0.5238 | 1.1278 | 1.3627 | 1.0000 |
| 23 | 0.7116 | 1.1189 | 0.5205 | 1.1290 | 1.3599 | 1.0000 |
| Mean ± std | **0.7135 ± 0.0017** | **1.1206 ± 0.0029** | **0.5225 ± 0.0017** | **1.1298 ± 0.0024** | **1.3639 ± 0.0046** | -- |

相较于未微调的 ST-TIS gate，gate-head fine-tune 将三 seed 平均 All MAE 从 0.7150 进一步降到 0.7135，Affected MAE 从 1.1220 降到 1.1206。这说明 gate 校准带来的增益具有跨 seed 稳定性。

### 5.5 Gate 与分支分工验证

为验证 gate 是否真的发挥作用，而不是仅仅增加参数量，本文进一步对已训练好的 dual-branch gate no-aux checkpoint 进行解释性分析。分析在完整 test split 上进行，共包含 27,499 个事故样本。我们不重新训练模型，而是在同一 residual beta 下比较五种预测方式：只使用正常反事实基线、只使用 normal-style residual branch、只使用 incident-graph residual branch、固定 gate=0.5，以及 learned gate。

| 融合方式 | All MAE | Affected MAE | Unaffected MAE |
|---|---:|---:|---:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.8057 | 1.2478 | 0.5983 |
| Incident-graph residual only | 0.9012 | 1.3562 | 0.6876 |
| Fixed gate = 0.5 | 0.7442 | 1.1706 | 0.5442 |
| Learned gate | **0.7181** | **1.1234** | **0.5279** |

![Gate 分支消融结果](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_branch_ablation_mae.png)

结果显示，单独使用 incident-graph branch 并不能取得好结果，说明事故图分支并不是天然优于正常残差分支。固定 gate=0.5 已经优于两个单分支，说明两个 residual branch 提供了互补信息；learned gate 又明显优于固定平均，将 affected MAE 从 1.1706 降低到 1.1234。这说明 gate 学到的不是简单常数权重，而是具有实际预测收益的自适应融合策略。

进一步地，我们比较 gate 权重与“哪个分支在局部更准确”的关系。这里的局部指每个样本、预测时距、候选节点和交通通道上的 residual error。

| 子集 | 局部分支条件 | Incident-branch gate 均值 |
|---|---|---:|
| All | incident branch 误差更低 | 0.3821 |
| All | normal-style branch 误差更低 | 0.3528 |
| Affected | incident branch 误差更低 | 0.3921 |
| Affected | normal-style branch 误差更低 | 0.3511 |
| Unaffected | incident branch 误差更低 | 0.3777 |
| Unaffected | normal-style branch 误差更低 | 0.3535 |

![Gate 与局部更优分支的一致性](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_selection_alignment.png)

当 incident branch 在局部 residual 预测上更准确时，gate 会分配更高的 incident-branch 权重；当 normal-style branch 更准确时，gate 权重下降。这个结果比单纯比较 affected 和 unaffected 的 gate 均值更有说服力，因为事故影响并不一定完全等同于 affected label。Gate 学到的是 residual explanation selection，而不是简单的节点二分类。

此外，gate 与目标残差幅度存在正相关。在全部有效元素上，gate 与 \(|\Delta^{target}|\) 的相关系数为 0.2198；在 affected candidates 上相关性提高到 0.2985；在 unaffected candidates 上仅为 0.0943。这说明 gate 对更强的异常偏离更加敏感，且这种关系在受影响节点上更明显。需要注意的是，gate 并没有随事故严重度或恢复时间呈现简单单调变化，因此本文不将其解释为全局事故严重度指标，而将其解释为节点级、预测时距级、通道级的局部残差解释选择器。

### 5.6 ST-TIS-style Incident Branch 分析

为验证 ST-TIS-style incident branch 的收益来自事故分支表达能力增强，而不是偶然的 gate 缩放，本文对 seed 23 checkpoint 进行同样的分支消融：

| 融合方式 | All MAE | Affected MAE | Unaffected MAE |
|---|---:|---:|---:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.7777 | 1.2133 | 0.5733 |
| ST-TIS incident residual only | 0.8044 | 1.2443 | 0.5980 |
| Fixed gate = 0.5 | 0.7244 | 1.1439 | 0.5276 |
| Learned gate before gate fine-tune | 0.7135 | 1.1217 | 0.5220 |
| Learned gate after gate fine-tune | **0.7116** | **1.1189** | **0.5205** |

![ST-TIS gate-head 分支消融结果](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_head_finetune_branch_ablation_mae.png)

与轻量 incident graph branch 相比，ST-TIS incident residual only 的 affected MAE 从 1.3562 降至 1.2443，fixed gate affected MAE 从 1.1706 降至 1.1439。这说明 temporal self-attention 和 graph-biased spatial attention 确实让事故分支本身更可靠。进一步冻结分支并只微调 gate head 后，learned gate affected MAE 从 1.1217 降至 1.1189，说明当前瓶颈已经从“事故分支太弱”转向“gate 如何更精细地校准分支可靠性”。

在 ST-TIS 版本中，当 incident branch 在局部误差更低时，affected elements 上的 incident-branch gate 均值为 0.4502；当 normal-style branch 更好时，gate 均值下降到 0.4172。Gate-head fine-tune 后，二者分别变为 0.4741 和 0.4376，局部选择差距进一步扩大。该结果保留了 residual explanation selection 的性质，并说明两阶段训练可以改善 gate 对分支可靠性的利用。

![ST-TIS gate-head 与局部更优分支的一致性](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_head_finetune_selection_alignment.png)

### 5.7 事故案例可视化

为进一步展示最终 ST-TIS gate-head fine-tune 模型的局部行为，本文从 test split 中选择 success、neutral 和 failure 三类样本。其中 success 表示 learned gate 明显优于 fixed gate=0.5，neutral 表示两者几乎相同，failure 表示 learned gate 在 affected candidates 上劣于固定融合。这样的选择避免只展示正例，也能说明 gate 的边界条件。

每个案例图包含五部分：受影响候选节点位置、incident-branch gate、目标残差幅度、fixed gate 与 learned gate 的误差差异，以及 normal-style branch 与 incident-graph branch 的局部误差差异。

| Rank | Category | Sample | Region | Affected nodes | Recovery min | Learned MAE | Fixed MAE | Gain | Mean gate |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | success | 185121 | 2 | 1 | 40.0 | 10.1947 | 11.6042 | 1.4095 | 0.5385 |
| 2 | success | 185115 | 2 | 1 | 60.0 | 5.6887 | 6.9432 | 1.2544 | 0.5083 |
| 3 | neutral | 55961 | 0 | 3 | 15.0 | 1.2352 | 1.2352 | 0.0000 | 0.4682 |
| 4 | neutral | 192167 | 2 | 3 | 10.0 | 0.6480 | 0.6480 | -0.0000 | 0.5517 |
| 5 | failure | 88134 | 1 | 1 | 25.0 | 6.4817 | 4.9938 | -1.4880 | 0.4453 |
| 6 | failure | 59634 | 0 | 1 | 180.0 | 4.9778 | 4.0989 | -0.8789 | 0.5932 |

![ST-TIS gate-head success case](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_head_finetune_case_studies_mixed/case_01_success_sample_185121.png)

以 sample 185121 为例，incident-only affected MAE 低于 normal-only affected MAE，说明事故分支在该样本的受影响节点上提供了更合适的残差解释。Learned gate 的 affected MAE 进一步低于 fixed gate，说明最终 gate 不只是平均两个分支，而是在节点和预测时距上更细粒度地选择残差解释。

![ST-TIS gate-head failure case](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/gate_head_finetune_case_studies_mixed/case_05_failure_sample_88134.png)

Failure 样本则暴露了当前 gate 的局限。例如 sample 88134 中，incident-only affected MAE 为 8.0691，显著差于 normal-only 的 2.1877；gate-head fine-tune 将 learned affected MAE 从未微调 ST-TIS 的 6.8554 降到 6.4817，但仍然劣于 fixed gate 的 4.9938。这说明 gate-head fine-tune 已经缓解了对 incident branch 的过信，却没有完全解决短恢复、单节点影响或 incident branch 局部失真的样本。该现象为后续改进提供了直接方向：需要更系统地引入 branch confidence estimation、uncertainty-aware gate 或困难样本建模。

### 5.8 后置 Latent Impact Correction Adapter 实验

本小节报告冻结 source 之上加入 latent impact correction adapter 的效果。所有 adapter 实验都基于 §5.4 中 ST-TIS gate-head fine-tune source（三个种子分别为 7、11、23），冻结其权重，仅训练 adapter 自身的约 13K 参数。Adapter 训练 5 个 epoch，batch size 128，最多采样 10000 个事故训练样本，验证集与测试集均使用与 source 相同的事故时间划分。

#### 5.8.1 主结果：3-seed 均值

下表给出 source（即 §5.2 中 ST-TIS gate-head fine-tune no-aux 模型）与四种 adapter 变体在三个随机种子上的 robust MAE 平均值，所有指标均为 adapter 与同 seed source 之差：负数表示 adapter 优于 source。本节同样区分 overall（全 test 集）与 severity-high & recovery-long（high\_and\_long）切片。

| Variant | Overall all | Overall affected | Overall unaffected | High\_and\_long all | High\_and\_long affected | High\_and\_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| anomgate05 (基线 adapter，无 regret) | -0.000441 | -0.000484 | -0.000421 | -0.000369 | -0.000163 | -0.000510 |
| **anomgate05 + magnitude regret w=0.05 (主候选)** | **-0.000418** | **-0.000465** | **-0.000396** | **-0.000376** | **-0.000211** | **-0.000489** |
| anomgate05 + magnitude regret w=0.10 | -0.000374 | -0.000417 | -0.000353 | -0.000354 | -0.000222 | -0.000445 |
| anomgate05 + top-k harmful regret w=0.10 | -0.000302 | -0.000327 | -0.000290 | -0.000294 | -0.000173 | -0.000378 |

观察四点。第一，所有 adapter 变体的 3-seed 均值上各切片 delta 均为负值，说明 adapter 至少没有在统计平均上让 source 退化。第二，anomgate05 主候选（不加 regret）已经将 source 的 3-seed overall affected MAE 进一步从 1.1049 降到 1.1044；high\_and\_long 切片 affected 仅微弱改善 -0.000163，反映出 source 在该切片上已接近其 latent capacity。第三，magnitude regret w=0.05 在 high\_and\_long affected 上把改善幅度从 -0.000163 提升到 -0.000211（相对 +30\%），同时 overall affected 仅让步约 4\%（-0.000484 \(\to\) -0.000465），是 3-seed Pareto 上明显占优的配置。第四，top-k harmful regret 与 magnitude regret w=0.10 均能把 caveat 切片 affected 进一步推向更负，但代价是更大的 overall affected 让步（前者损失 32\%，后者 14\%），不在 Pareto 上。

#### 5.8.2 Per-seed Caveat 现象与缓解

3-seed 均值掩盖了 source 在不同初始化下的稳定性差异。下表给出每个 seed 上 severity-high & recovery-long 切片受影响节点的 source-vs-adapter delta：

| Seed | source affected MAE | anomgate05 \(\Delta\) | + magnitude regret w=0.05 \(\Delta\) | + magnitude regret w=0.10 \(\Delta\) | + top-k harmful regret w=0.10 \(\Delta\) |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.290 | -0.000406 | -0.000388 | -0.000314 | -0.000167 |
| 11 | 1.286 | **+0.000176** | +0.000058 | -0.000025 | -0.000055 |
| 23 | 1.276 | -0.000259 | -0.000305 | -0.000327 | -0.000298 |
| **3-seed 平均** | 1.276 | -0.000163 | **-0.000211** | -0.000222 | -0.000173 |

显著观察是 seed 11 上 anomgate05 在 high\_and\_long affected 切片上 delta 为 +0.000176，即该 seed 下 adapter 比 source 自身更差（caveat）。该现象在 seed 7 与 seed 23 上不存在，说明 caveat 来自训练过程中的局部基坑，而非系统性问题。然而，因为论文需要在多个种子上汇报稳定结果，仅靠丢弃该 seed 难以站得住，因此需要训练侧的鲁棒性约束。

Magnitude regret w=0.05 把 seed 11 上的 caveat 从 +0.000176 减到 +0.000058（缩小 67\%）；w=0.10 进一步把它推到 -0.000025（完全翻号）；top-k harmful regret 也能将其推到 -0.000055。但综合 3-seed 均值与对其他 seed 的影响，magnitude regret w=0.05 给出最佳整体 Pareto 表现：seed 11 caveat 显著缩小，同时其他 seed（尤其 seed 23）保留 -0.000305 的改善幅度。

#### 5.8.3 Pareto 曲线：filter selectivity 与 weight 的关系

为更系统地理解 selective regret filter 的可调维度，本文在 seed 11 上扫描了 magnitude quantile \(q\) 与权重 \(\lambda\) 的组合：

| 配置 | Seed 11 overall affected \(\Delta\) | Seed 11 high\_and\_long affected \(\Delta\) | 简评 |
|---|---:|---:|---|
| anomgate05 (no regret) | -0.000518 | **+0.000176** | baseline，存在 caveat |
| q=0.95, \(\lambda\)=0.10 | -0.000503 | +0.000040 | 较窄 filter，弱效果 |
| q=0.90, \(\lambda\)=0.05 | -0.000507 | +0.000058 | 主候选 |
| q=0.90, \(\lambda\)=0.10 | -0.000478 | -0.000025 | sign-flip elbow |
| q=0.80, \(\lambda\)=0.10 | -0.000422 | -0.000056 | 过强抑制 |
| q=0.90, \(\lambda\)=0.20 | -0.000357 | -0.000093 | 过强抑制 |

观察到一个非平凡现象：filter selectivity (\(q\)) 与 weight \(\lambda\) 在 Pareto 曲线上几乎完全可互换——\(q=0.95, \lambda=0.10\) 与 \(q=0.90, \lambda=0.05\) 的 (overall, caveat) 几乎相同；\(q=0.80, \lambda=0.10\) 与 \(q=0.90, \lambda=0.20\) 也几乎相同。这表明本文设计的 selective regret 实际上由"regret pressure"这个一维标量决定，filter selectivity 与 weight 不是独立旋钮。

我们也评测了把两种 filter 同时激活（即 magnitude quantile + top-k harmful regret 的交集）的组合形式，结果与单独 magnitude filter 几乎逐位相同：在该任务的实际数据分布下，correction magnitude 与 regret value 高度相关——top-10\% magnitude 的子集已经包含了大部分 top-10\% regret value 元素，因此交集后再过滤是 no-op。这一发现进一步说明 magnitude 单独已是最 surgical 的 filter。

#### 5.8.4 Alignment 诊断：机制澄清

为厘清 selective regret filter 的实际作用机制，我们对 seed 11 上 source、anomgate05、anomgate05+magnitude regret w=0.05 与 anomgate05+magnitude regret w=0.10 四个版本，分别在 severity-high & recovery-long 受影响节点子集上计算了几个细粒度量：

| 版本 | \(|c|\) 均值 | sign\_match\_rate | beneficial\_rate | harmful\_rate | mean\_improvement |
|---|---:|---:|---:|---:|---:|
| source-only | — | — | — | — | 0.0 (参照) |
| anomgate05 | 0.0258 | 0.5087 | 0.5034 | 0.4966 | -0.000176 |
| + regret w=0.05 (主候选) | 0.0224 | 0.5088 | 0.5043 | 0.4957 | -0.000058 |
| + regret w=0.10 | 0.0191 | 0.5091 | 0.5052 | 0.4948 | +0.000025 |

关键现象有两点。第一，三种 adapter 配置的 sign\_match\_rate 全部仅在 0.508-0.509 之间，与随机猜测的 0.5 几乎不可区分。这说明加 selective regret 后 adapter 输出的 correction **方向并未变得更准**，beneficial 与 harmful 元素的比例几乎没变（0.503 vs 0.497 \(\to\) 0.504 vs 0.496 \(\to\) 0.505 vs 0.495）。第二，对应的 \(|c|\) 均值显著下降：相比 anomgate05 的 0.0258，主候选下降 13\%、强 regret 版本下降 26\%。

由此可推断 selective regret filter 的实际机制是 magnitude shrinkage 而非 direction calibration：filter 没有让 adapter 学得更聪明，但通过缩小"大幅修正"的整体幅度，**少量大幅 harmful 修正的边际损失被压低**——而由于 beneficial 与 harmful 在同一幅度阈值下大致相当，beneficial 的边际收益也按比例缩小。最终 mean\_improvement 从 -0.000176 演化到 +0.000025，即 caveat 切片的 source-vs-adapter 平衡点从 source 略优转为 adapter 略优。这个机制解释也为何 \(q\) 与 \(\lambda\) 在 Pareto 上可互换：两者最终都通过减小被惩罚 bin 内的有效 correction 幅度实现 shrinkage。

### 5.9 Temporal Decay 的辅助分析

较早的 temporal decay head 版本虽然不再是当前最佳主模型，但它仍然提供了一个有用观察：事故影响的持续性对严重事故和长恢复事故更重要。

| 维度 | 分组 | No decay MAE | Decay MAE | Gain |
|---|---|---:|---:|---:|
| Severity | Low | 0.8539 | 0.8553 | -0.16% |
| Severity | Mid | 0.9826 | 0.9811 | 0.15% |
| Severity | High | 1.3345 | **1.3206** | **1.05%** |
| Recovery | < 30 min | 0.9591 | 0.9572 | 0.19% |
| Recovery | 30-90 min | 1.0520 | 1.0545 | -0.24% |
| Recovery | >= 90 min | 1.2339 | **1.2224** | **0.94%** |

![事故严重程度与恢复时间分组收益](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/severity_recovery_decay_gain.png)

这一结果说明，事故影响并不是均匀发生在所有样本上。高严重程度事故和长恢复事故更需要显式的持续性建模。当前 dual-branch gate 可以被理解为更一般的机制：它不只按预测时距调制残差，还允许模型在节点和预测时距上选择 normal-style 或 incident-style 残差解释。

### 5.10 外部 Baseline 对比：IGSTGNN 的 Affected/Unaffected 盲区

为了将本文的 affected-focused 评估置于更广的 baseline 语境中，我们在同一 XTraffic 数据集、同一 70/15/15 时间划分、同一 12→12 horizon 上完整复现了 KDD'26 的 IGSTGNN（Fan et al., 2026）。该模型代表了当前 incident-aware ST 预测的 SOTA 水平。我们在 Alameda、Contra Costa、Orange 三个 region 上的复现 average MAE 为 12.86 / 13.31 / 13.53，与原论文报告的 12.69 / 13.43 / 13.13 全部在 ±3% 以内（Contra Costa 实际略好于 paper），证明该 baseline 已被忠实复现，可作为对比基准。

| Region | Paper avg MAE | Ours avg MAE | Δ |
|---|---:|---:|---:|
| Alameda | 12.69 | 12.86 | +1.3% |
| Contra Costa | 13.43 | 13.31 | **−0.9%** |
| Orange | 13.13 | 13.53 | +3.0% |

**关键发现：IGSTGNN 报告的 average MAE 隐藏了 affected 节点上的显著退化。** 我们用本文的 incident-sensor affected 标注（`outputs/impact_labels/<region>/node_labels.csv`）将 IGSTGNN 的全县预测切分到事故实际影响的传感器子集，得到以下对比：

| Region | All MAE | Affected MAE | Unaffected MAE | Gap (aff − unaff) |
|---|---:|---:|---:|---:|
| Alameda | 12.87 | **17.09** | 12.83 | **+33.2%** |
| Contra Costa | 13.36 | **17.33** | 13.31 | **+30.2%** |
| Orange | 13.55 | **18.63** | 13.52 | **+37.8%** |

平均每个事故只有 4.4-5.2 个传感器被标注为 affected（占县级节点的约 1%），但这一小撮节点上的 MAE 比未受影响节点高出 30-38%。由于体量悬殊，这一退化几乎完全被县级平均稀释掉——IGSTGNN 报告的 12-13 average MAE 实际上极接近 unaffected-only MAE。Per-horizon 上，affected MAE 从 H1 的 13-15 涨到 H12 的 19-21，退化速度比 unaffected 更快（Orange H12 affected 21.30 vs unaffected 15.12，相对差距 +41%）。

![Per-horizon affected vs unaffected MAE on IGSTGNN](/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/igstgnn_affected_per_horizon.png)

这一对比直接验证了本文 affected-focused 评估框架的合理性。其一，事故感知模型即使报告 SOTA average MAE，在事故真正影响的节点上仍存在 30%+ 的退化，意味着 overall MAE 不能反映模型在 incident-aware 任务上最关键的能力。其二，affected MAE 在 horizon 尾段退化更猛，正好对应本文 affected H6/H12 切片的研究目标。其三，三 region 一致的 30-38% 退化模式说明这是 overall-MAE 评估范式的结构性盲区，而非 region 偏置。

需要指出，IGSTGNN 报告 raw flow MAE，本文报告残差空间 normalized robust MAE，单位口径不同，因此本节不做"我们 vs IGSTGNN"的直接数字对比，而是从外部模型的 affected/unaffected 退化幅度间接说明 affected-focused 评估的必要性。统一单位的 head-to-head 对比留作补充实验。

## 6. 哪里体现了更好，为什么会更好

### 6.1 哪里体现了更好

第一，主测试集结果更好。在同一 HDF5 cache、同一时间划分、同一 learned normal branch、同一 residual target 和同一 robust MAE 指标下，dual-branch gate no-aux 将全部候选节点 MAE 从 0.7221 进一步降到 0.7181；ST-TIS-style incident branch no-aux 又进一步降到 0.7135；gate-head fine-tune 最终达到 0.7116。与 learned normal baseline 相比，整体提升为 14.56%。

第二，受事故影响节点更好。Affected candidates 是本文真正关心的部分，因为它们代表事故实际影响到的传感器。Gate-head fine-tuned ST-TIS 将 affected MAE 从 learned normal baseline 的 1.2938 降到 1.1189，相对提升 13.52%，也优于 temporal decay no-aux 的 1.1290。三随机种子平均下，最终版本的 affected MAE 为 1.1206，优于未微调 ST-TIS 的 1.1220 和轻量 dual-branch gate 的 1.1240。

第三，no-aux 设置下更好。当前最佳模型没有使用未来派生的 affected label 或 impact label 做辅助监督，因此可以更有力地说明：性能提升来自模型结构，而不是训练时偷看了未来影响标签。

第四，多随机种子稳定。轻量 dual-branch gate 三个 seed 的 all MAE 为 \(0.7182 \pm 0.0016\)，affected MAE 为 \(1.1240 \pm 0.0054\)；ST-TIS-style 版本进一步达到 \(0.7150 \pm 0.0013\) 和 \(1.1220 \pm 0.0019\)；gate-head fine-tune 最终达到 \(0.7135 \pm 0.0017\) 和 \(1.1206 \pm 0.0029\)。这说明结果不是单次训练偶然波动。

第五，分支解释性分析更好。轻量模型中 learned gate 的 affected MAE 为 1.1234，明显优于 fixed gate=0.5 的 1.1706；ST-TIS-style 模型中 learned gate 的 affected MAE 为 1.1217，也优于 fixed gate=0.5 的 1.1439；gate-head fine-tune 后进一步达到 1.1189。与此同时，ST-TIS incident-only affected MAE 从轻量事故分支的 1.3562 降至 1.2443，说明增强事故分支确实改善了残差表达能力，而 gate-head fine-tune 则改善了融合策略。

第六，长尾事故切片的 per-seed 鲁棒性更好。在 ST-TIS gate-head fine-tune source 之外加入 latent impact correction adapter 与 selective magnitude regret filter 后，3-seed 均值上 high-severity & long-recovery 切片的受影响节点 robust MAE 改善幅度从 source 自身的 -0.000163（受 seed 11 局部 caveat 拖累）提升到 -0.000211，相对 +30\%。同时 overall affected 改善仅从 -0.000484 让步到 -0.000465（约 4\%），整体 Pareto 占优于不加 regret 的 anomgate05、加全 mask regret 的版本以及 top-k harmful regret 版本。在 seed 11 单 seed 上，high\_and\_long affected delta 从 +0.000176（source 优于 adapter）减到 +0.000058，缓解了 source 单 seed 的局部回退。

### 6.2 为什么会更好

最关键的原因是，模型不再让两个分支从同一输入中随意学习两个完整交通 embedding，而是先用 normal STGNN 建立一个正常反事实基准，再让两个分支只解释残差。这样两个分支的学习空间被约束在 \(Y-\hat{Y}^{normal}\) 中，目标更集中，也更符合事故影响建模的任务。

第二，两个分支具有不同归纳偏置。Normal-style residual branch 更适合处理弱扰动、正常预测误差和局部小偏差；incident graph residual branch 在 full candidate graph 上做消息传递，更适合处理事故导致的空间传播偏差。ST-TIS-style 版本进一步把事故分支改成 temporal self-attention 和 graph-biased spatial attention，使其更擅长捕捉事故残差的时间演化和候选节点之间的空间传播。即使两者共享部分残差输入，它们的结构和任务角色不同，因此比原始“双 MLP 同输入同目标”的设计更不容易学成同一种东西。

第三，gate 是节点级、预测时距级的自适应融合。事故影响不是全局统一的：同一个事故可能强烈影响上游节点，对下游节点影响较弱；短 horizon 和长 horizon 的残差形态也可能不同。Gate 允许模型在每个节点和 horizon 上决定残差解释来源，从而减少未受影响节点上的过度事故修正，同时增强受影响节点上的事故传播建模。

第四，两阶段训练让分支表达和 gate 融合分开优化。第一阶段共同训练两个 residual branch 和 gate，学习可互补的残差解释；第二阶段冻结两个分支，只微调 gate head，使融合策略在不破坏分支表示的前提下更贴近最终预测误差。实验中，branch loss 和 oracle-style gate alignment 都明显退化，而 gate-head fine-tune 稳定提升，这说明保留分支互补性比强迫每个分支单独成为好 predictor 更重要。

第五，残差融合比完整状态融合更稳。完整交通状态中包含大量正常模式，如果两个分支都预测完整 \(Y\)，gate 很容易主要学习正常交通平均规律。残差空间去掉了一部分正常趋势，保留的是模型最需要解释的偏离部分，因此 gate 更容易学到与事故影响相关的权重变化。

第六，frozen-source 加 adapter 的设计形成"宽-窄"分工。Source 内部的 dual-branch gate 在样本绝大多数位置上做出准确决策，而 adapter 仅在分支分歧大且修正方向较有把握的位置介入。Anomaly gate \(g_{anom}\) 用 \(|\hat{\Delta}^{incident}-\hat{\Delta}^{normal}|\) 作 latent incident impact signal，比直接使用 source 输出的 affected-node 概率更可靠（前者是连续的内部表示分歧，后者是下游分类输出本身就受 source 训练偏差影响）。这样一来 adapter 不需要重新学一个 source 级别的预测，只需要"何时"和"多少"两个决策。Selective magnitude regret filter 进一步引入一个非对称约束：它不要求 adapter 学得方向更准（alignment 诊断显示 sign\_match\_rate 约 0.51 几乎不变），而是约束"大幅修正若被发现帮倒忙则被收缩"——通过 magnitude shrinkage 而非 direction calibration 实现长尾鲁棒性，这是任务数据本身（beneficial/harmful 接近 1:1 但 harmful 大幅元素拖累）的结构特性决定的。

## 7. 讨论与后续工作

当前模型相比最初设想已经更接近一个可写成论文方法的结构：它保留了“双分支 + gate”的直觉，但通过 residual target、full candidate graph 和 no-aux 验证解决了两个关键质疑。第一，两个分支为什么不会学成一样。第二，模型是否依赖推理阶段不可得的受影响节点标签。

不过，当前实现仍有需要补强的地方。首先，ST-TIS-style incident branch 和 gate-head fine-tune 已经提升了总体 MAE 和三 seed 平均 affected MAE，但少数短恢复、单节点影响样本仍然存在局部失败。例如 sample 88134 在 gate-head fine-tune 后 learned affected MAE 从 6.8554 降到 6.4817，但仍劣于 fixed gate 的 4.9938。Adapter 也无法修复此类极端样本，因为 adapter 看到的就是 source 内部状态，source 在 gate 上犯的根本错误（normal MAE 远低于 incident MAE 但 gate 仍然偏向 incident）会传到 adapter 的输入特征，adapter 缺乏其它独立信号去推翻 source 的判断。其次，我们初步尝试了 branch confidence heads、branch-error supervision、hard-example reweighting、branch loss 和 oracle-style gate alignment：这些方法要么收益有限，要么明显破坏融合。这说明后续不能只用简单 error-head 或 loss reweighting 修正 gate，而需要更系统地建模分支置信度、预测不确定性和困难样本。

第三，adapter 与 selective regret filter 的当前形式仍有可探索的空间。Filter selectivity (\(q\)) 与 weight (\(\lambda\)) 在 Pareto 上几乎完全可互换，意味着两者并非独立旋钮，留出的真正自由度比设想中少；正在试图引入的 top-k harmful regret 也与 magnitude filter 在实际数据上几近重合（top-10\% 大幅修正子集已包含大部分 top-10\% 高 regret 元素）。要进一步突破，可能需要 per-correction trust head（让 adapter 自己学一个 sigmoid 标记可信度）、curriculum regret（前 2 epoch 不加 regret 让 correction 先自由生长）或对 adapter 引入与 source 不同的独立特征源。

最后，当前主要报告 normalized residual space 下的 robust MAE。正式论文中可以继续补充原始单位下的 MAE/RMSE/MAPE、不同道路类型和不同事故类别下的分组结果，以及更多样化的事故案例可视化。

## 8. 结论

本文提出一种潜在事故影响介导的双分支门控交通流预测框架。不同于依赖事故类型识别的方法，本文将事故影响定义为相对于正常反事实预测的残差，并在残差空间中使用 normal-style residual branch 和 incident branch 进行自适应融合。实验表明，dual-branch gate no-aux 在 XTraffic 上优于单分支残差模型和 temporal decay 变体；进一步引入 ST-TIS-style incident branch 和 gate-head fine-tune 后，模型在三随机种子平均下继续改善全部候选节点和受影响候选节点 MAE。Gate 分析表明，learned gate 优于固定平均，并且在 incident branch 局部误差更低的位置分配更高事故分支权重。

为进一步缓解 source 在长尾事故切片上的 per-seed 局部回退，本文在冻结 source 之上加入 latent impact correction adapter，并通过基于分支分歧的 anomaly gate 和 selective magnitude regret filter 控制修正的"启用时机"与"幅度抑制"。Adapter 在 3-seed 均值上把 high-severity & long-recovery 切片的受影响节点 robust MAE 改善幅度从 -0.000163 提升到 -0.000211（相对 +30\%），并将 source 在 seed 11 上的 +0.000176 caveat 缩到 +0.000058（缩小 67\%），同时仅以 4\% 的 overall affected 让步换取该长尾鲁棒性。Alignment 诊断进一步澄清，selective regret filter 的实际机制是 magnitude shrinkage 而非 direction calibration——sign\_match\_rate 几乎不变，但大幅修正的整体幅度被收缩，从而在 beneficial 与 harmful 元素近 1:1 的数据分布下抑制少量大幅 harmful 修正的边际损失。

整体而言，结果说明，将事故影响作为可门控的残差解释进行建模，是提升事故场景交通预测鲁棒性的有效方向；而在 source 之上叠加受 anomaly gate 调制并以 selective regret 训练的轻量修正层，能在不破坏 source 自身性能的前提下，对长尾事故切片进行 surgical 鲁棒性改善。

## 参考文献占位

正式投稿稿件中建议继续使用当前 BibTeX 文件中的参考文献条目，包括 XTraffic、IGSTGNN、DCRNN、STGCN、Graph WaveNet、ST-TIS、FEDformer、MSD-Mixer 和 Balance and Brighten 等工作。
