# DCCP 分支标签有效性审计

这个独立实验回答：VLA 提出的同一组候选动作，经世界模型预测和 Progress LRM 产生的 DCCP 偏好标签，是否得到真实 Coffee 模拟器结果支持。

实验代码只在 `dccp/experiments/branch_validity_audit/` 中。它不修改、不导入 trainer 入口，不创建 optimizer，也不启动 Ray/FSDP。现有 `state_selection_audit` 只作为库和真实可恢复状态来源。

## 数据流与防泄漏门禁

```text
真实可恢复状态
  → 冻结 8 个 VLA candidate（candidate 0 是 nominal）
  → 世界模型生成预测视频 → Progress LRM 打分
  → 按当前 DCCP nominal-vs-alternative margin 规则冻结标签
  → 之后才允许模拟器真实执行
  → 计算 pair、top-1、ranking、regret、校准指标
```

`labels/LABELS_FROZEN.json` 保存标签文件 SHA-256。模拟器阶段会重新验证哈希。若冻结前已经存在真实模拟器结果，冻结阶段会拒绝继续，必须换 `run_name`。

当前 DCCP 标签不是简单的全局最高分对最低分。每个 alternative 都与 nominal 比较：

- `alternative - nominal > margin_pos`：alternative 胜；
- `alternative - nominal < -margin_neg`：nominal 胜；
- 其余比较不产生训练标签。

审计同时保留全局预测 top/bottom 作为诊断。

## 前置步骤

先完成真实状态收集和状态选择：

```bash
cd /path/to/DCCP/dccp/experiments/state_selection_audit
./run_coffee_smoke.sh setup
./run_coffee_smoke.sh restore
./run_coffee_smoke.sh collect
./run_coffee_smoke.sh select
./run_coffee_smoke.sh inspect
```

默认读取 `/path/to/DCCP/dccp/outputs/state_selection_audit/coffee_smoke_v1`。也可在配置中手动指定：

```yaml
states:
  max_states: 1
  targets:
    - trajectory_id: trajectory_000
      decision_index: 12
```

## 推荐的第一次 tiny 运行

每个阶段结束都会自动重建 `<output>/<run_name>/inspection/index.html`。

```bash
cd /path/to/DCCP/dccp/experiments/branch_validity_audit
export AUDIT_CONFIG=$PWD/configs/coffee_tiny.yaml

./run_coffee.sh setup
./run_coffee.sh states
./run_coffee.sh inspect
./run_coffee.sh candidates
./run_coffee.sh inspect

./run_coffee.sh wm --max-predictions 1
./run_coffee.sh inspect
./run_coffee.sh wm

./run_coffee.sh freeze --score-only
./run_coffee.sh inspect
./run_coffee.sh freeze

./run_coffee.sh simulator --max-rollouts 1
./run_coffee.sh inspect
./run_coffee.sh simulator

./run_coffee.sh metrics
./run_coffee.sh inspect
```

tiny 为 `1 state × 2 candidates × 1 WM seed × 1 simulator seed × H=1`。GPU 被占用时，`candidates`、`wm` 和 `simulator` 会打印 `SKIPPED`，不加载模型、不写失败记录。

tiny 通过后改用 `configs/coffee_smoke.yaml`：2 个状态、每状态 8 个候选、3 个 WM seed、3 个模拟器 seed、H=3。重型阶段支持断点续跑和精确过滤：

```bash
./run_coffee.sh wm --state-id TRAJ__step_0012 --candidate-index 3 --wm-seed 401
./run_coffee.sh simulator --state-id TRAJ__step_0012 --candidate-index 3 --evaluation-seed 101
```

冻结后不能重评分；改变候选、种子、阈值、WM 模式或 LRM 规则时必须换 `run_name`。

## 每一步看什么

1. `states`：起始画面、决策位置、状态来源和哈希。
2. `candidates`：候选第一个 7 维动作；0 号是否为 nominal。
3. `wm`：是否从同一画面出发；物体/机械臂是否漂移；候选差异是否合理。
4. `freeze`：每个 WM seed 的 LRM 分数、均值、top/bottom 和实际产生的 pairs。
5. `simulator`：真实视频、相同后续 VLA seed、Coffee 阶段指标和失败原因。
6. `metrics`：pair 准确率、success lift、失败转成功、top-1、regret、相关性和校准。

主要标准答案是模拟器最终成功率。短期标准答案来自 Coffee 独立的 `grasp/rim/insertion/task` 指标，映射为 `0.4/0.6/0.8/1.0`；不会用同一个 LRM 给真实视频打分来充当标准答案。

## 世界模型随机性

- `controlled`：同一状态和 WM seed 下所有候选共享逐步 diffusion noise，是主要因果比较。
- `faithful`：候选使用独立 diffusion noise，复现当前在线 rollout 的噪声行为。

`faithful` 只复现噪声语义。在线训练的 nominal suffix 可能来自缓存 imagined rollout，而真实状态没有那个缓存 suffix；审计会从同一真实画面对 nominal 也重新生成，报告中必须说明。

默认 `label_aggregation: primary_seed` 使用预注册的第一个 WM seed 产生正式标签，复现一次在线 DCCP 随机分支；其余 seed 用来报告稳定性。只有显式选择 `mean` 才审计多 seed 集成标签。

## 报告与指标

产物包括 `validity_report.json`、三份 CSV 和 `inspection/index.html`。核心指标：

- `strict_pair_accuracy_all_pairs`（真实 tie 算不正确）和 `decisive_pair_accuracy`；
- `pair_nonworse_rate`、`winner_loser_success_lift`、`fail_to_success_rate`；
- tie-aware top-1、success regret、Spearman/Kendall；
- 预测 margin 与真实 success gap 的关系及分箱校准。

本实验验证真实可达、可恢复 Coffee 状态上的标签；无法为世界模型想象出的不可恢复状态提供 MuJoCo 标准答案。候选应来自未参与训练的轨迹，manifest 会记录源路径和哈希，但数据划分证明仍需随最终报告保存。

## CPU 测试

```bash
/path/to/conda_envs/wmpo_env/bin/python -m pytest -q \
  /path/to/DCCP/dccp/experiments/branch_validity_audit/tests
```
