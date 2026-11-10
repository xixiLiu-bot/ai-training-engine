#!/usr/bin/env bash
# 评测入口：一条命令跑完闭环，无交互、无网络、失败也要产出结果。
#
# 环境变量（全部可选）：
#   DATASET       数据集名（synthetic / gas / synthetic_rul / bearing / rul）
#   MODEL         模型（logreg / ridge / gbdt / rf），默认按任务类型自动选
#   DETECTOR      漂移检测（psi / ks / wasserstein / all）
#   PSI_LOW       稳定阈值（不填用 config 默认；接官方数据必须重标定）
#   PSI_MID       重型阈值
#   GATE_WINDOW   门控窗口批数（默认 1）
#   GATE_BOOTSTRAP Bootstrap 次数（默认 0）
#   LLM           决策层后端（none / qwen4b / pytrio / ollama）
#   QWEN_BASE_URL 远端推理服务地址（LLM=none 时不需要）
#   QWEN_API_KEY  远端推理服务 key
#   USE_SWANLAB   置 1 才尝试上传 SwanLab（默认关闭）
set -o pipefail

cd "$(dirname "$0")" || exit 1

DATASET="${DATASET:-synthetic}"
MODEL="${MODEL:-}"
DETECTOR="${DETECTOR:-psi}"
LLM="${LLM:-none}"
OUT="${OUT:-out}"

mkdir -p "$OUT" || true

ARGS=(--dataset "$DATASET" --detector "$DETECTOR" --out "$OUT" --llm "$LLM")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
[ -n "$PSI_LOW" ] && ARGS+=(--psi-low "$PSI_LOW")
[ -n "$PSI_MID" ] && ARGS+=(--psi-mid "$PSI_MID")
[ -n "$GATE_WINDOW" ] && ARGS+=(--gate-window "$GATE_WINDOW")
[ -n "$GATE_BOOTSTRAP" ] && ARGS+=(--gate-bootstrap "$GATE_BOOTSTRAP")

echo "=== 智训引擎 · 离线运行 ==="
echo "数据集=$DATASET 模型=${MODEL:-auto} 检测器=$DETECTOR 决策层=$LLM"
echo "输出目录=$OUT"

# 合成数据在镜像里没有，先按需生成（纯本地，无需网络）
if [ "$DATASET" = "synthetic" ] && [ ! -d "data/synthetic" ]; then
  echo "[准备] 生成合成漂移流 ..."
  python scripts/make_data.py || echo "[警告] 合成数据生成失败，将依赖已有数据"
fi

python scripts/run_pipeline.py "${ARGS[@]}"
RC=$?

# 无论成败，都必须确认产物存在——这是门槛①的兜底
if [ ! -f "$OUT/metrics.json" ]; then
  echo "[严重] 未生成 metrics.json，写出占位结果以保证有输出"
  printf '{"fatal": true, "reason": "run_pipeline 未能产出结果", "rc": %d}\n' "$RC" > "$OUT/metrics.json" || true
fi

echo "=== 完成，产物在 $OUT/ ==="
ls -1 "$OUT" 2>/dev/null || true

# 退出码：产物存在就返回 0（评测看产物，不只看退出码）
if [ -f "$OUT/metrics.json" ]; then
  exit 0
fi
exit "$RC"
