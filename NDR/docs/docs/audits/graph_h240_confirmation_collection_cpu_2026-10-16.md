# Graph H240 的 Mac 消费者 CPU 联接验收

本次只增加临时机械 fixture 测试及本说明。生产 controller、collector、模型、trainer、统计规则和原 37 项测试均保持提交 `7f3e94fd` 的原字节。原独审回执 `004e1cf0` 继续保留。

## 新增验证范围

新增测试实际执行原 `collect_stage`，通过冻结 df38 transport 的逐文件 local-reuse 分支复制全部声明文件，再执行原 `admit_copied` 和 `family6` 本体。没有替换 `validate_plan`、`validate_inventory`、原等待证据校验、native receipt 校验、树哈希或 family 消费者。

真实 seed42 完整包只读复用。独立 schema 复核已确认完整 148 文件树、a6 回执、62cd 分析、两份 source frame、两项 pooled config、verification、观测集合和共同数据统计身份符合消费者字段。独审 schema 回执 SHA 为 `66e3bd6552f98fdf83a5becd09d36432b7c9cd14f3739f2af92771570fbd4c40`。

新 seeds1、2 的八项完成记录、native receipt 和 OS 声明全部由 fixture 手工构造。这些内容明确标记为 CPU 机械数据，不是真实 source、Torch、Windows Handle 或训练证据。checkpoint 是带有 `NOT_PICKLE_NOT_TRAINED` 标识的普通字节，Mac 没有反序列化它们。

为了验证真实 `family6` 本体的接受路径，fixture 使用生产形状的 foreign-Windows plan 和 receipt。该 plan 只存在于 pytest 临时树，外层 scope 明确声明没有生产 admission、部署、实际预算或科学结果。所有固定生产门禁保持原样。另一个测试保留 `synthetic_only=True`，确认科学出口依旧拒绝。

## 针对检查

共八项新增检查：

1. 完整 opaque 收集、逐文件 transport manifest、原终态验证和六模型 family 本体的 Mac 接受路径。
2. seed2 的单地区失败，结果保持负向，其他地区和种子不能抵消。
3. `synthetic_only=True` 仍然拒绝 family 输出。
4. 第二份 source frame 改变，不提升目录，不生成保全回执。
5. 损坏的映射字节在本 CPU 测试中由网络 trap 阻断，不提升目录。原生产 transport 的合法网络 fallback 没有改动。
6. 保全后的 checkpoint 字节改变，family 在输出比较结果前拒绝。
7. 所有 opaque 文件、native receipt 和 source frame 摘要重新绑定后，原 held Handle wait 为 `null` 仍被真实校验器拒绝。
8. 索引已有代码档案前先识别 foreign-Windows 标签，确认不会把这些 provenance 标签交给本机 Path I/O。

测试对 Popen、远端 connection.python、Torch、研究模型模块和 HDF5 导入设置显式 trap。所有收集请求均须使用完整逐文件 SHA-256 local-reuse，临时输出不进入正式结果目录。缺少私有 source43、seed42 或冻结 transport 档案时相关测试明确跳过。

检查入口：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/ndr_research/test_graph_h240_confirmation_collection.py --disable-warnings
```

## 仍未验证的范围

本次证明 Mac 消费者联接能够接受和拒绝预期的机械元数据，不构成 Windows 八项真实监督或源端原生 checkpoint 审计，也不构成科学家族结果。临时文件即使具有科学消费者的状态字段，也只属于明确标记的 pytest fixture。

真实 cost、strong4 前驱、seeds1 和 2 训练、Windows 原 Handle 事实、真实完整成功或失败保全与科学 family6 仍须后续实际验收。`production_launchable=false` 保持。没有 SSH、GPU、HDF5 数据、Orange 或 real test 访问，没有新真实执行计划，也没有改变用户保留游戏的状态。
