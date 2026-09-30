# 实验协议与不可变约束

## 审计对象

时间单位是一次 VLA 决策，即一个 `8 × 7` 动作块的起点，不是动作块内部的单个 robosuite step。对 nominal 轨迹的每个决策起点保存相机图像、MuJoCo flattened state、model XML、原动作、归一化动作、response token、动作 token 熵和累计底层步数。

Progress LRM 对从决策 `t` 开始、长度为 `1 + H` 个决策帧的 nominal suffix 打分。默认 `H=3`。曲率只在 progress 序列的内部位置有效，因此首尾点不进入任何方法、random 或 oracle 的比较域。

## 选择冻结

在 simulator branch 产生前同时计算：

1. random：在满足预算和 NMS 的全部合法组合中均匀抽样；
2. curvature：`|q[t+1] - 2q[t] + q[t-1]|`；
3. entropy：冻结 VLA 在 256 个动作 token 子词表上的平均 masked entropy；
4. joint：轨迹内 min-max 归一化后的曲率与熵加权和。

三种打分方法复用 `verl/utils/dccp_mining.py` 的 local maxima 和 temporal NMS。选择文件原子写入后生成冻结标记。已有 branch 结果时禁止首次选择或强制重算。

## 候选动作与真实回放

每个状态的候选集合只生成一次并落盘：candidate 0 必须是 nominal 原动作，其余候选由同一冻结 VLA 采样并按 response token 去重。候选集合不能按评估 seed 改变。

每个 `(轨迹, 状态, 候选, seed)` 使用新建环境：

1. `reset_to(model_xml, flattened_state)`；
2. 执行候选的完整首动作块；
3. 后续仍由同一冻结 VLA 控制；
4. 成功或达到和 nominal 相同的全局底层步上限时停止。

同一 `(轨迹, 状态, evaluation seed)` 下，各候选后续第 `k` 次 VLA 采样使用相同派生 seed。这是 common random numbers，只减少候选比较方差，不让选择器看到结果。

## 指标

```text
p(t, a) = 同一组 evaluation seeds 上的成功均值
improvement(t) = max_a p(t, a) - p(t, nominal)
```

改善量被限制为非负，因为候选集合包含 nominal。失败转成功必须是配对事件：同一个 seed 上 nominal 失败且至少一个候选成功。

每种方法报告选中状态的平均改善量、失败转成功机会/配对转化率，以及相对同预算/NMS oracle 的捕获比例。random 是大量合法盲抽的期望。oracle 只在最终统计阶段读取改善量，不能用于训练或候选生成。

## 有效性检查

- 状态恢复闸门必须先通过。
- nominal 采集代码会读取恢复报告并强制执行该闸门，而不是只依赖文档约定。
- 所有方法共享完全相同的有效状态域。
- 汇总前必须具有每个有效状态的全部候选和全部 seed；`FAILED` 不能当作失败回报。
- 要改变超参数请使用新的 `run_name`，不要改写已经冻结的选择。
- tiny 仅验证软件。5 条轨迹 pilot 只用于发现效应和估计成本；正式结论应增加轨迹和 seed，并报告区间。

## 人工可视化验收

`inspect` 在任意阶段生成离线 HTML，不调用模型或模拟器。它包含 nominal contact sheet、选择曲线/时间轴、方法选择表、阶段完成度、branch 结果与可选 MP4、最终汇总表。

第一次真实执行必须先用过滤参数只跑一个 branch，并保存视频。确认恢复起点、候选首动作、冻结 VLA 后续控制和终止条件都正确后，才扩大到单状态和全部有效状态。正式全量遍历默认不录视频，避免无意产生数千个大文件；需要审计的分支通过过滤器选择性录制。
