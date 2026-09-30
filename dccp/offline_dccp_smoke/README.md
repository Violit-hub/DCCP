# DCCP 离线分阶段 Smoke

这个目录放 **独立验证脚本**，不修改 WMPO / verl 主训练流程。目标是在单卡显存不够时，把原本在线同时占显存的流程拆开：

```text
在线 full smoke 当前会同时占用：
policy / ref / world model / progress LRM / DCCP branch 临时张量
```

这里先提供一个最小可运行的 staged smoke：

```text
阶段 1：生成一个 toy branch artifact
阶段 2：只启动 progress LRM，对 artifact 里的 nominal / alternative branch videos 打分，并打包 pref_* batch
阶段 3：关闭 LRM，只读取 pref_* batch，验证 actor 的 DPO-style preference loss 接口
```

注意：`make_toy_branch_artifact.py` 当前用 toy videos 模拟 world model 已经生成好的 nominal / counterfactual branch。它的 artifact 格式就是后续真实 world-model dump 应该保存的最小字段。

## 为什么先做 toy artifact

真正的单卡分阶段 full flow 需要把现有在线 rollout worker 拆成可恢复 artifact：

```text
vla_history
nominal imagined videos
selected states
candidate action tokens
counterfactual branch videos
policy context tensors
```

这会牵涉到 `robwm_rollout.py` 内部对象序列化。为避免污染主训练流程，本目录先把 **LRM scoring 与 pref_* 打包、B 部分 loss 消费** 跑通。

## 运行顺序

### 1. 生成 toy artifact

不需要 GPU，不需要 LRM：

```bash
cd /path/to/DCCP/dccp
conda activate wmpo_env
python offline_dccp_smoke/make_toy_branch_artifact.py \
  --output /tmp/dccp_offline/artifact.pt
```

### 2. 启动 progress LRM

另开一个终端：

```bash
cd /path/to/DCCP/dccp
conda activate vlm_reward
CUDA_VISIBLE_DEVICES=0 bash reward_model/lrm_server/start_progress_server.sh
```

确认 `/progress` 可访问后，在 `wmpo_env` 终端打分：

```bash
cd /path/to/DCCP/dccp
conda activate wmpo_env
python offline_dccp_smoke/score_artifact_with_progress_lrm.py \
  --artifact /tmp/dccp_offline/artifact.pt \
  --output /tmp/dccp_offline/pref_batch.pt \
  --progress-endpoint http://127.0.0.1:8002/progress \
  --delta-plus 0.02 \
  --delta-minus 0.02
```

打完分后可以关掉 progress LRM，释放显存。

### 3. 验证 pref_* loss 消费

不需要 LRM；默认使用 fake actor logprob，不加载 OpenVLA 大模型：

```bash
cd /path/to/DCCP/dccp
conda activate wmpo_env
python offline_dccp_smoke/check_pref_loss_artifact.py \
  --pref-batch /tmp/dccp_offline/pref_batch.pt \
  --use-ref-gap false
```

如果输出包含：

```text
loss/dccp_pref = ...
dccp/valid_pairs = ...
```

说明 packed pref_* batch 能被 B 部分 loss 接口消费。

## 和论文链路的对应关系

这个 staged smoke 覆盖：

```text
branch progress scoring
→ margin 判断 winner-loser
→ packed pref_* schema
→ DPO-style local preference loss
```

toy artifact 暂不覆盖真实：

```text
当前策略 + world model 生成 nominal imagined trajectories
progress curvature + entropy 的真实 selected states
真实 counterfactual branch rollout
```

这些可以在后续把 `make_toy_branch_artifact.py` 替换为真实 `dump_world_model_branch_artifact.py` 后接上。


A1. policy + world model 生成 nominal trajectory，保存 suffix videos
A2. 关 world model，开 progress LRM，给 suffix videos 打分，选 selected states
A3. 关 LRM，重新开 policy + world model，从 selected states 生成 branch videos
A4. 关 world model，开 progress LRM，给 branch videos 打分
A5. 关 LRM，开 actor/ref 做 loss