#!/usr/bin/env bash
# 清理废弃"事故导向"实验的死方向 outputs 产物 (未跟踪的大 checkpoint / 转换数据 / 缓存).
# 用 git clean -xdf: 只删未跟踪 + 被 gitignore 忽略的文件.
# 所有 git 跟踪的实验小结 (config.json / summary.md / metrics.csv 等) 一律自动保留,
# 包括 outputs/igstgnn_ours_pipeline_fixed/setting_a_breakdown.json.
#
# 用法:
#   bash scripts/cleanup_dead_outputs.sh          # dry-run, 只打印将删除的内容 (默认)
#   bash scripts/cleanup_dead_outputs.sh --apply   # 真正删除
#
# 引用核对: 下列目录在 CLAUDE.md / XTraffic项目完整汇报.md 里引用计数均为 0.
# 活证据 (diagnostics / baselines / rgdn / paper_artifacts / dist_net/region_data) 不在此列表.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DIRS=(
  outputs/igstgnn_data_from_ours              # 31G   IGSTGNN 转换输入 (可由 export_to_igstgnn.py 再生)
  outputs/impact_guided_next_stage            # 12G   事故导向双支实验 checkpoint (跟踪的小结保留)
  outputs/full_candidate_stgnn_heatmap_model  # 9.3G  候选热图 STGNN 消融 checkpoint
  outputs/igstgnn_ours_pipeline_fixed         # 1.9G  旧管线 (保留跟踪的 setting_a_breakdown.json)
  outputs/igstgnn_ours_pipeline               # 630M  旧管线
  outputs/impact_labels                       # 389M  事故标签中间产物
  outputs/impact_labels_fixcheck              # 192M
  outputs/impact_labels_aggregated            # 111M
  outputs/impact_labels_test                  # 8.1M
  outputs/forecast_error_validation           # 15M
  outputs/fourier_test                        # 7.3M
  outputs/impact_case_plots                   # 4.4M
  outputs/impact_case_plots_fixcheck          # 0B
  outputs/candidate_residual_model            # 4.1M
  outputs/candidate_stgnn_residual_model      # 1.5M
  outputs/impact_residual_model               # 1.0M
  outputs/impact_quality                      # 520K
)

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

if [[ $APPLY -eq 0 ]]; then
  echo "=== DRY-RUN (未删除任何文件) ==="
else
  echo "=== APPLY (真正删除) ==="
fi

before=$(du -sk . 2>/dev/null | cut -f1)

for d in "${DIRS[@]}"; do
  [[ -e "$d" ]] || { echo "skip (不存在): $d"; continue; }
  if [[ $APPLY -eq 1 ]]; then
    git clean -xdf "$d"
  else
    git clean -xdfn "$d"
  fi
done

if [[ $APPLY -eq 1 ]]; then
  after=$(du -sk . 2>/dev/null | cut -f1)
  freed=$(( (before - after) / 1024 / 1024 ))
  echo ""
  echo "已回收约 ${freed} GB. 剩余仓库大小: $(du -sh . 2>/dev/null | cut -f1)"
  echo "git 跟踪证据检查 (应为空, 无跟踪文件被删):"
  git status --short
else
  echo ""
  echo "以上为 dry-run 预览. 确认无误后执行:"
  echo "  bash scripts/cleanup_dead_outputs.sh --apply"
fi
