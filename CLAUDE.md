# XTraffic 交通流量预测研究 — 项目状态与约定

详细记录见《项目总结报告_2026-06-10.md》(单一事实来源,含全部数字与出处)。
本文件只做快速导航,新会话先读这里。

## 一句话现状 (2026-06-18,RGDN 架构方向负结果,回 XTraffic 应用论文主线)

**老师新方向 RGDN(去季节化残差引导双支)已试,负结果,机制不成立。** 分解导向:复用缓存
气候态基线 baseline_median[day_kind,tod] 去季节化、只预测偏离;残差支线图传播,主支线节点局部
加邻居残差注入,sum 融合。Alameda 种子 42 等参数消融(全部对齐 ~316k):v0a 原始 11.811 /
v0b 去季节 11.711 / v1 完整 11.884 / v2 无注入 11.779。**去季节化是唯一真增益**:v0b 比 v0a
all −0.100,单 GWN 就够、不需要双支。**机制不立**:v1 比 v0b 差 +0.174;注入是负担,v1 比 v2
差 +0.105;双支本身 v2 vs v0b +0.068 在噪声带内。**注入是优化陷阱**(诊断坐实):自适应邻接
退化成全局平均、熵比 0.987,模型重度依赖、推理置零崩到 22.6,门控版 v1g 11.922 更差且学到的
门控停在 0.90、逃不出。数据 outputs/diagnostics/rgdn_round1_results.txt;代码 fourier_dual_net/
rgdn.py + scripts/{train_rgdn,collect_rgdn_results,diagnose_rgdn_injection}.py;spec/plan 见
docs/superpowers。单种子,方向清楚(headline ~2-4σ),多种子可收紧但结论已定。
**下一步:与老师对齐,把 RGDN 负结果 + 去季节化小增益并入 XTraffic 应用论文,或决定补多种子坐实。**

下面是 RGDN 之前的主线状态(仍有效)。证据链完整,桌面有完整汇报
`~/Desktop/XTraffic项目完整汇报_2026-06-17.md`(11 节,所有数字核自落盘)。
**论文定位 = XTraffic 应用/复现性研究**:命题=事故标签无增益。最强证据=STAEformer
(ICCV'23 SOTA, label-free, 1.56M 参数) 在 XTraffic 上 Alameda 11.39 / CC 12.12,
**超过用全套标签的 IGSTGNN**,也超过我们的 FDN(11.98)。Orange STAEformer 后台重跑中
(990 节点慢,首次因网络切换中断)。
IGSTGNN 原论文(论文/2602.02528v1.pdf)三漏洞已核:基线过弱(没放 STAEformer)、
无去标签对照(我们 ICSF 移植零增益证伪)、代码 bug 让主表存疑。
**两个待批准的决定性实验(等老师拍板):** (1) 全滑窗协议复跑直接对标其 Table 4 的 12.69;
(2) STAEformer ± ICSF 标签注入,证"连最强模型加标签也零增益"。
频谱路由是标注边界的分析点,不是主菜(等参数下容量占 74-80%,路由仅 0.7%)。

**频谱路由的诚实结论**(经等参数消融定案,勿再夸大):基准上 FDN 赢 GWN 的旧 headline
(PEMS04 −0.69 / PEMS08 −0.51) **是 2P-vs-P 不公平对比**(FDN=2×GWN+7 参数)。等参数三变体
(gwn/dual/spectral) 拆开:容量占 74-80%,路由仅 20-26%。路由净效应 pooled −0.137 MAE
(6/6 种子负, t=−5.8),真实但小(~0.7%)。STID(有 time emb)路由=0,METR-LA(speed)=0。
=> 频谱路由 = 轻量周期注入,仅在「周期主导信号 + 周期盲骨干」交集有效。
数据:outputs/diagnostics/{gwn_routing_vs_capacity, spectral_stid_ablation, benchmark_transfer_results}.txt。
教训:任何"X 赢 Y"先查参数是否对齐,再下结论(14.3 节改正)。
**下一步:与老师对齐选刊与篇幅,然后开写 XTraffic 论文。**

## 已定结论(不要重新讨论)

1. FDN (FFT 可学习掩码 + 双 GWN) 赢 GWN 6-9σ (Alameda -0.38, CC -0.42, 3 种子),
   赢 IGSTGNN (matched windows, 用标签) Alameda+Orange 显著、CC 平。
2. 事故标签无增益:ICSF/TIID 移植到 GWN 零增益 (12 runs);IGSTGNN 唯一赢的格子
   (CC 碰撞窗口) 与其事故模块无关;hazard 窗口占 42-58% 稀释一切。
3. IGSTGNN 官方 dataloader bug:num_threads = max(bs//2,1) 硬编码 → 无条件触发,
   有效 batch=2。我们的对比用修复版(对其有利)。
4. 架构增强 5 连败:+A/+B/+E/D3 gated fusion/ICSF 移植,全部在种子噪声内。
   不要再提调制模块类方案。
5. fixed_k → learnable 掩码 +0.225 (Alameda, >5σ):这是真实的架构元素,论文卖点之一。
6. RGDN 残差引导双支 (老师新方向, 2026-06-18) 负结果:去季节化 (单 GWN 预测偏离气候态)
   是唯一真增益 (v0b−v0a all −0.10, 等参数);双支无增益 (v2≈v0b);注入是优化陷阱、门控也
   救不回 (v1g 11.922 > v2 11.779, gate 停 0.90)。架构增强连败再+1,别再提注入/双支调制类。

## 下一步任务:标准基准移植(已批准,进行中)

- 数据:**已下载到 data/benchmarks/**(gitignore,勿提交):metr-la.h5 54M +
  adj_mx_metrla.pkl(3 元组,最后一项 207×207 邻接);PEMS04.npz (16992,307,3) +
  PEMS04_distance.csv (from,to,cost);PEMS08.npz (17856,170,3) + PEMS08_distance.csv。
  PEMS-BAY 暂缓(GitHub raw 100M 限制,3 个数据集足够判读)。
- 协议:12 步入 12 步出;METR-LA 用 speed 通道 70/10/20,值为 0 视为缺失做 masked MAE;
  PEMS04/08 用 flow 通道 0,60/20/20,全部有效。z-score 用训练段统计。
- 实现:新建 scripts/train_benchmark.py:通用 loader(滑窗采样,无事件锚定概念)+
  复用 FDN 与 GWN 模型构造;--model {fdn,gwn} --dataset {metrla,pems04,pems08};
  时间特征 tod 从时间索引推(METR-LA 5min 起点 2012-03-01 00:00,PEMS 5min);
  FDN 的 time_feat 需要 (B,T_h,2) [tod/288, dow/7] 同我们管线。各 3 种子;
  数据 scp 到 5080 的 data/benchmarks/ 后用隐藏任务队列跑,每 run <1h。
- 判读:FDN 对 GWN 的种子带差距是否复现 0.3+ 量级;赢 → 方法论文,
  XTraffic 标签分析降为一章;不赢 → 基准研究论文。
- 然后与老师对齐(带报告)。

## 关键文件

- 项目总结报告_2026-06-10.md — 全部结果、13 节审计实验、14 节最新决策
- scripts/train_fourier_dual_net.py / train_graphwavenet.py / train_stid.py /
  train_gwn_icsf.py — 四个训练脚本,产物 schema 相同
- scripts/significance_tests.py / incident_type_breakdown.py / diagnose_fdn_failures.py
- outputs/diagnostics/ — seed_noise_band.txt, icsf_collision_results.txt, significance_tests.txt
- outputs/dist_net/region_data/ — XTraffic h5 缓存 (本地+5080 同构)
- docs/superpowers/specs+plans — D3 与 ICSF 的设计文档
- 本地无 torch;一切训练在 5080。

## 5080 远程操作手册(踩坑后的最终版)

- LAN: asus@192.168.31.13, 密码 Hzj050916;Tailscale 100.126.189.30 时常不在线。
- 代码在 C:/Users/asus/traffic_fourier;python 在 cmd PATH,torch 在 user-site。
- 长训练启动:写 .bat 队列 (CRLF!) + VBS 隐藏包装 + schtasks 创建/run/确认日志后
  **立即删除任务**(占位触发时间会真的再次发火,血的教训:三实例抢 GPU)。
- 不要用: nohup(bash 无 python)、Start-Process(引号地狱)、/ru SYSTEM 或
  /ru asus /rp(批处理登录无 user-site → 找不到 torch)、裸控制台(被关 → forrtl 200)。
- 读远端日志:python -X utf8 + base64 经 ssh 管道,避免 GBK 乱码;杀队列要连父 cmd
  树一起 taskkill /T,否则 cmd 前进到下一条命令继续拉起 python。
- Monitor 过滤词至少含: test MAE|Traceback|forrtl|aborting|CUDA out of memory。

## 写作与协作约定

- 用户要诚实的负结果;所有汇报数字必须出自已落盘文件,严禁估算填充。
- 报告类文稿:不用破折号、不用括号内插说明(全局 CLAUDE.md 的规则)。
- 代码注释最少化;commit 信息英文;种子约定 42/1/2;指标 masked MAE all/affected/unaffected。
