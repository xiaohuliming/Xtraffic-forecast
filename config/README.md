# 实验室运行配置

`lab_hosts.env` 是实验室共享连接参数的唯一仓库内来源。脚本通过 Bash `source` 读取它，各变量均采用“已有环境变量优先”的默认值写法。

## 使用

```bash
source config/lab_hosts.env
ssh "${LAB_5080_SSH}"
ssh -p "${LAB_SEETACLOUD_PORT}" "${LAB_SEETACLOUD_SSH}"
```

`scripts/watch_zoo_queue.sh` 会默认读取这个文件。调用时可以直接覆盖单个运行参数：

```bash
RELAY=other@host RELAY_DIR=/other/path \
  bash scripts/watch_zoo_queue.sh
```

也可以切换整份配置：

```bash
LAB_HOSTS_CONFIG=/path/to/another.env \
  bash scripts/watch_zoo_queue.sh
```

SeetaCloud 密码随实例变化，因此 `LAB_SEETACLOUD_PASSWORD` 默认留空。实例启动后可在当前 shell 中设置。
