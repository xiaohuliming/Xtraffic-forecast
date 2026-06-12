# XTraffic 交通流量预测研究 — 项目状态与约定

详细记录见《项目总结报告_2026-06-10.md》(单一事实来源,含全部数字与出处)。
本文件只做快速导航,新会话先读这里。

## 一句话现状 (2026-06-13)

label-free 的 FourierDualNet (FDN) 是 XTraffic 上的 SOTA;事故标签被证明无增益;
5 次架构增强全部无增益;审计实验全部关账。**进行中的下一步:把 FDN 移植到标准基准
(METR-LA / PEMS-BAY / PEMS04/08) 验证频谱路由增益是否可迁移 — 结果决定论文定位**
(赢 → 方法论文 "learnable spectral routing";不赢 → XTraffic 基准研究论文)。

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
