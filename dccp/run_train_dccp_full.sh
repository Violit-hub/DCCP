#!/usr/bin/env bash
set -euo pipefail

DCCP_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCCP_ROOT="$(cd "$DCCP_CODE_ROOT/.." && pwd)"
export DCCP_ROOT
DCCP_ASSET_ROOT="${DCCP_ASSET_ROOT:-$DCCP_ROOT/assets}"
export DCCP_ASSET_ROOT
cd "$DCCP_CODE_ROOT"

STOP_RAY_BEFORE_START=${STOP_RAY_BEFORE_START:-0}
if [[ "$STOP_RAY_BEFORE_START" == "1" ]]; then
  ray stop || true
fi

export PYTHONPATH=$PWD:$PWD/dependencies/opensora:$PWD/dependencies/openvla-oft:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# LRM 服务通过端口转发进来时，训练脚本只使用本机可见的训练 GPU。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
N_GPUS=${N_GPUS:-1}
export N_GPUS
PYTHON_BIN=${PYTHON_BIN:-python}

# 默认值用于先确认完整跑通一轮 PPO 更新；正式跑大配置时从命令行覆盖。
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
N_SAMPLES=${N_SAMPLES:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
TRAJ_MINI_BATCH_SIZE=${TRAJ_MINI_BATCH_SIZE:-1}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-999}
SAMPLE_NUM=${SAMPLE_NUM:-1}
SAVE_FREQ=${SAVE_FREQ:-2}
TEST_FREQ=${TEST_FREQ:-999999}
SKIP_FINAL_VALIDATION=${SKIP_FINAL_VALIDATION:-true}

DCCP_HORIZON_H=${DCCP_HORIZON_H:-3}
DCCP_BRANCH_HORIZON=${DCCP_BRANCH_HORIZON:-3}
DCCP_STATE_BUDGET_PER_TRAJ=${DCCP_STATE_BUDGET_PER_TRAJ:-2}
DCCP_SELECTED_STATES=${DCCP_SELECTED_STATES:-2}
DCCP_NUM_CANDIDATES=${DCCP_NUM_CANDIDATES:-8}
DCCP_MAX_PAIRS_PER_ROLLOUT=${DCCP_MAX_PAIRS_PER_ROLLOUT:-4}
DCCP_MAX_PAIRS_PER_BATCH=${DCCP_MAX_PAIRS_PER_BATCH:-64}
DCCP_PREF_MICRO_BATCH_SIZE=${DCCP_PREF_MICRO_BATCH_SIZE:-1}
DCCP_LAMBDA_PREF=${DCCP_LAMBDA_PREF:-0.3}
DCCP_LAMBDA_PREF_WARMUP_STEPS=${DCCP_LAMBDA_PREF_WARMUP_STEPS:-0}
DCCP_COMPUTE_ACTION_ENTROPY=${DCCP_COMPUTE_ACTION_ENTROPY:-true}
DCCP_REQUIRE_ENTROPY=${DCCP_REQUIRE_ENTROPY:-true}
DCCP_ALLOW_MISSING_ENTROPY_FALLBACK=${DCCP_ALLOW_MISSING_ENTROPY_FALLBACK:-false}
DCCP_ACTION_VOCAB_SIZE=${DCCP_ACTION_VOCAB_SIZE:-256}
DCCP_VERBOSE_LOG=${DCCP_VERBOSE_LOG:-1}

# 正式多轮训练默认只保存 LoRA adapter，避免中间 checkpoint merge 全模型时额外吃显存。
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-false}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}

# ===== 需要按自己的数据集修改的变量 =====
POLICY=${POLICY:-$DCCP_ASSET_ROOT/policies/coffee}
STAT=${STAT:-$POLICY/dataset_statistics.json}
WM_CFG=${WM_CFG:-$DCCP_ROOT/assets/world_model_configs/coffee/inference_cfg.py}
STATE_PATH=${STATE_PATH:-$DCCP_CODE_ROOT/verl/utils/dataset/back/coffee_d0_states.pkl}
DATA_FILES_ROOT=${DATA_FILES_ROOT:-$DCCP_ROOT/assets/coffee/data_files}
UNNORM_KEY=${UNNORM_KEY:-coffee_d0_300_demos}
TASK_NAME=${TASK_NAME:-coffee}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-coffee}
CAMERA=${CAMERA:-cam_left_wrist}
# 只给 LRM scorer 用；为空时 rollout 会回退到 VLA 的 task_description。
LRM_TASK_DESCRIPTION=${LRM_TASK_DESCRIPTION:-"Task: Place the small upright white cylindrical object from the right side of the table into the red cup under the coffee machine, then close the gray lid over the cup.
Progress stages:
- Low progress: the robot has not approached or grasped the white cylinder.
- Partial progress: the robot approaches or grasps the white cylinder.
- Medium progress: the white cylinder is being moved toward the red cup.
- High progress: the white cylinder has been placed fully inside the red cup.
- Complete: the white cylinder is inside the red cup and the gray lid is fully closed over the cup.
The task progress should decrease if the white cylindrical object slips from the gripper, falls onto the table, is dropped outside the red cup, or is only partially inserted and then falls out."}

COMPLETION_ENDPOINT=${COMPLETION_ENDPOINT:-http://127.0.0.1:8001/completion}
PROGRESS_ENDPOINT=${PROGRESS_ENDPOINT:-http://127.0.0.1:8002/progress}
COMPLETION_HEALTH_ENDPOINT=${COMPLETION_HEALTH_ENDPOINT:-http://127.0.0.1:8001/health}
PROGRESS_HEALTH_ENDPOINT=${PROGRESS_HEALTH_ENDPOINT:-http://127.0.0.1:8002/health}
PRECHECK_LRM=${PRECHECK_LRM:-1}

LOG_DIR=${LOG_DIR:-$DCCP_CODE_ROOT/logs}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/dccp_ray}
RAY_OBJECT_SPILLING_DIR=${RAY_OBJECT_SPILLING_DIR:-$RAY_TMPDIR/spill}
mkdir -p "$LOG_DIR" "$RAY_TMPDIR" "$RAY_OBJECT_SPILLING_DIR"
export RAY_TMPDIR
export RAY_object_spilling_directory=$RAY_OBJECT_SPILLING_DIR

LOG=$LOG_DIR/train_dccp_full_$(date +%Y%m%d_%H%M%S).log
DCCP_VERBOSE_JSONL=${DCCP_VERBOSE_JSONL:-$LOG_DIR/dccp_verbose_$(date +%Y%m%d_%H%M%S).jsonl}
export DCCP_VERBOSE_LOG
export DCCP_VERBOSE_JSONL

for path in "$POLICY" "$STAT" "$WM_CFG" "$STATE_PATH" "$DATA_FILES_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[preflight] missing required path: $path" >&2
    exit 1
  fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[preflight] python not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "[preflight] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_GPUS=$N_GPUS TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE N_SAMPLES=$N_SAMPLES"
echo "[preflight] DCCP_VERBOSE_LOG=$DCCP_VERBOSE_LOG DCCP_VERBOSE_JSONL=$DCCP_VERBOSE_JSONL"
"$PYTHON_BIN" - <<'PYTORCH_CUDA_CHECK'
import os
import sys
import torch

expected = int(os.environ.get("N_GPUS", "1"))
count = torch.cuda.device_count()
print(
    f"[preflight] torch.cuda.is_available={torch.cuda.is_available()} "
    f"device_count={count} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    flush=True,
)
if not torch.cuda.is_available() or count < expected:
    print(f"[preflight] need {expected} visible CUDA device(s), got {count}", file=sys.stderr, flush=True)
    sys.exit(1)
PYTORCH_CUDA_CHECK

check_lrm_endpoint() {
  local name="$1"
  local health_url="$2"
  local fallback_url="$3"
  echo "[preflight] checking ${name}: ${health_url}"
  if curl --noproxy '*' -fsS --max-time 10 "$health_url" >/dev/null; then
    echo "[preflight] ${name} health endpoint is reachable"
    return 0
  fi
  echo "[preflight] ${name} health failed; trying endpoint GET: ${fallback_url}"
  curl --noproxy '*' -fsS --max-time 10 "$fallback_url" >/dev/null
  echo "[preflight] ${name} endpoint is reachable"
}

if [[ "$PRECHECK_LRM" == "1" ]]; then
  check_lrm_endpoint "completion server" "$COMPLETION_HEALTH_ENDPOINT" "$COMPLETION_ENDPOINT"
  check_lrm_endpoint "progress server" "$PROGRESS_HEALTH_ENDPOINT" "$PROGRESS_ENDPOINT"
fi

DCCP_MARGIN_POS=${DCCP_MARGIN_POS:-0.1}
DCCP_MARGIN_NEG=${DCCP_MARGIN_NEG:-0.1}
DCCP_BETA=${DCCP_BETA:-0.1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-coffee_dccp}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$DCCP_CODE_ROOT/checkpoints/$EXPERIMENT_NAME}
export MUJOCO_GL=${MUJOCO_GL:-egl}

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  trainer.project_name=DCCP \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_local_dir=$CHECKPOINT_DIR" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.val_before_train=false \
  trainer.val_only=false \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  ++trainer.skip_final_validation=$SKIP_FINAL_VALIDATION \
  'trainer.logger=[console]' \
  ++trainer.wandb_mode=disabled \
  algorithm.adv_estimator=grpo \
  reward_model.enable=false \
  verifier.reward_coef=5 \
  use_dccp_branch=true \
  actor_rollout_ref.use_dccp_branch=true \
  actor_rollout_ref.actor.use_dccp_branch=true \
  actor_rollout_ref.ref.use_dccp_branch=true \
  actor_rollout_ref.rollout.use_dccp_branch=true \
  scorer.completion_endpoint=$COMPLETION_ENDPOINT \
  scorer.progress_endpoint=$PROGRESS_ENDPOINT \
  scorer.timeout_sec=300 \
  lrm_input.camera=$CAMERA \
  "++lrm_input.task_description='${LRM_TASK_DESCRIPTION}'" \
  "++actor_rollout_ref.rollout.lrm_input.task_description='${LRM_TASK_DESCRIPTION}'" \
  dccp.horizon_H=$DCCP_HORIZON_H \
  dccp.branch_horizon=$DCCP_BRANCH_HORIZON \
  dccp.state_budget_per_traj=$DCCP_STATE_BUDGET_PER_TRAJ \
  dccp.selected_states=$DCCP_SELECTED_STATES \
  dccp.num_candidates=$DCCP_NUM_CANDIDATES \
  dccp.max_pairs_per_rollout=$DCCP_MAX_PAIRS_PER_ROLLOUT \
  dccp.max_pairs_per_batch=$DCCP_MAX_PAIRS_PER_BATCH \
  +dccp.pref_micro_batch_size=$DCCP_PREF_MICRO_BATCH_SIZE \
  dccp.margin_pos=$DCCP_MARGIN_POS \
  dccp.margin_neg=$DCCP_MARGIN_NEG \
  dccp.delta_plus=$DCCP_MARGIN_POS \
  dccp.delta_minus=$DCCP_MARGIN_NEG \
  dccp.compute_action_entropy=$DCCP_COMPUTE_ACTION_ENTROPY \
  dccp.require_entropy=$DCCP_REQUIRE_ENTROPY \
  dccp.allow_missing_entropy_fallback=$DCCP_ALLOW_MISSING_ENTROPY_FALLBACK \
  +dccp.action_vocab_size=$DCCP_ACTION_VOCAB_SIZE \
  dccp.beta=$DCCP_BETA \
  dccp.lambda_pref=$DCCP_LAMBDA_PREF \
  dccp.lambda_pref_warmup_steps=$DCCP_LAMBDA_PREF_WARMUP_STEPS \
  actor_rollout_ref.wm.enable=true \
  actor_rollout_ref.wm.inference_config_path=$WM_CFG \
  ++actor_rollout_ref.wm.build_legacy_reward_model=false \
  actor_rollout_ref.model.path=$POLICY \
  actor_rollout_ref.model.dataset_statistics_path=$STAT \
  actor_rollout_ref.model.lora_rank=$LORA_RANK \
  actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
  ++actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
  ++actor_rollout_ref.model.save_merged_model=$SAVE_MERGED_MODEL \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.pretrained_checkpoint=$POLICY \
  actor_rollout_ref.rollout.unnorm_key=$UNNORM_KEY \
  actor_rollout_ref.rollout.task_suite_name=$TASK_SUITE_NAME \
  ++actor_rollout_ref.rollout.data_files_root=$DATA_FILES_ROOT \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.val_micro_batch_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.traj_mini_batch_size=$TRAJ_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  ++actor_rollout_ref.ref.vla=openvla-oft \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_response_length=56 \
  data.n_samples=$N_SAMPLES \
  data.sample_num=$SAMPLE_NUM \
  data.task_suite_name=$TASK_SUITE_NAME \
  data.task_name=$TASK_NAME \
  data.state_path=$STATE_PATH \
  data.rollout_batch_size=$ROLLOUT_BATCH_SIZE \
  "$@" \
  2>&1 | tee "$LOG"

echo "log saved to: $LOG"
