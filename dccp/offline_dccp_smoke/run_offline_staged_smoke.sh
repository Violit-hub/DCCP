#!/usr/bin/env bash
set -euo pipefail

# DCCP 单卡离线分阶段 smoke。
# 这个脚本不启动 LRM；它只负责阶段 1 和阶段 3。
# 阶段 2 需要你手动启动 progress server 后运行 score_artifact_with_progress_lrm.py，
# 因为这样才能在打完分后手动关闭 LRM、释放显存。

cd /path/to/DCCP/dccp

OUT_DIR=${OUT_DIR:-/tmp/dccp_offline}
mkdir -p "$OUT_DIR"

ARTIFACT=${ARTIFACT:-$OUT_DIR/artifact.pt}
PREF_BATCH=${PREF_BATCH:-$OUT_DIR/pref_batch.pt}

python offline_dccp_smoke/make_toy_branch_artifact.py --output "$ARTIFACT"

cat <<EOF

[offline-dccp] 阶段 1 完成：$ARTIFACT

接下来请另开终端启动 progress LRM：

  cd /path/to/DCCP/dccp
  conda activate vlm_reward
  CUDA_VISIBLE_DEVICES=0 bash reward_model/lrm_server/start_progress_server.sh

然后在当前 wmpo_env 终端运行阶段 2：

  python offline_dccp_smoke/score_artifact_with_progress_lrm.py \\
    --artifact "$ARTIFACT" \\
    --output "$PREF_BATCH" \\
    --progress-endpoint http://127.0.0.1:8002/progress

阶段 2 完成后，关闭 progress LRM，再运行阶段 3：

  python offline_dccp_smoke/check_pref_loss_artifact.py \\
    --pref-batch "$PREF_BATCH" \\
    --use-ref-gap false

EOF
