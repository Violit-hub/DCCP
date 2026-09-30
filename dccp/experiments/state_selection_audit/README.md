# Coffee 状态选择审计实验

这个目录实现一个与训练流程完全分离的小规模实验：检查曲率、动作不确定性及二者联合选择的两个时刻，是否比随机时刻具有更大的真实动作改善空间。

本实现只新增 `dccp/experiments/state_selection_audit/`。它不修改 `verl`、训练配置、`install.sh` 或已有 checkpoint；策略只以 `eval()`/`inference_mode()` 加载，不创建 optimizer。真实标签来自 robosuite/MuJoCo 分支回放，不使用世界模型。

## 两套规模

- `configs/coffee_tiny.yaml`：1 条轨迹、2 个候选动作、1 个评估 seed、最多 64 个底层步。只验证管线，不能报告科学结论。
- `configs/coffee_smoke.yaml`：5 条轨迹、每状态 8 个候选（第 0 个是原动作）、3 个评估 seed、最多 256 个底层步。这是第一轮小规模 pilot。

完整 pilot 的成本仍然不小。若每条轨迹有 32 个 VLA 决策，`H=3` 后约有 27 个曲率有效状态，分支上界约为 `5 × 27 × 8 × 3 = 3240` 次真实 rollout。建议先跑 tiny，检查报告和状态产物，再启动 pilot。

## 固定状态、固定候选的 25-seed 稳定性复验

第一轮 `coffee_one_traj_v2` 完成后，可固定其 Simulator Oracle 状态 3、6
及每个状态的 8 个候选动作，仅用 25 个全新配对 seed 重新评估：

```bash
cd /path/to/DCCP/dccp/experiments/state_selection_audit
./run_coffee_fixed_oracle_25seeds.sh start
./run_coffee_fixed_oracle_25seeds.sh status
./run_coffee_fixed_oracle_25seeds.sh log
```

新结果写入 `coffee_fixed_oracle_s3_s6_25seeds_v1/`，不会写入或复制旧实验的
`branch_results.jsonl`。准备阶段会复制并校验冻结候选的 SHA-256；评估共
`2 states × 8 candidates × 25 seeds = 400` 个分支。该复验是对已发现敏感状态的
post-hoc 稳定性检查，不应作为无偏的状态选择结果。本实验不需要 Progress LRM。

## 为什么单独目录是合适的

实验通过 adapter 只读复用以下稳定接口：

- `robomimic` 环境的 `get_state()` / `reset_to()`；
- OpenVLA-OFT 的 `generate_action_verl()`；
- DCCP 已有的 progress 曲率、归一化、local maxima 和 temporal NMS；
- DCCP 的 256 个 action-token masked entropy。

运行入口、配置、产物 schema、统计和测试都留在本目录。训练代码升级时只需检查 adapter，不会把一次性审计逻辑混入 DCCP 训练主循环。

## 环境与路径

VLA 和模拟器阶段固定使用：

```bash
/path/to/conda_envs/dccp_env/bin/python
```

代码从 `/path/to/DCCP/dccp` 导入，模型、数据、MuJoCo 和输出路径都写在 YAML 中。入口会设置 MuJoCo、NVIDIA 动态库和 `PYTHONPATH`，无需修改 `.bashrc`，也不会改变 `dccp_env` 中的 `-e` 安装。

初始状态是项目已有的 pickle。配置中的 `trust_initial_states_pickle: true` 只表示信任这个确定路径；不要替换成来源未知的 pickle。

## 单轨迹后台自动完成

`coffee_one_traj.yaml` 使用 1 条完整轨迹、8 个候选动作和 3 个配对评估种子。完成首次视频闸门检查后，可让总控在后台自动完成剩余阶段：

```bash
cd /path/to/DCCP/dccp/experiments/state_selection_audit
./run_coffee_one_traj_full.sh start
```

总控会读取已有产物并断点续跑，不会重算 `branch_results.jsonl` 中已经成功的主键。GPU 忙时自动等待；evaluate 每 8 条形成一个小块，小块结束后重新检查状态。达到完整分支数后会自动执行 `summarize` 和 `inspect`。

查看进度或日志：

```bash
./run_coffee_one_traj_full.sh status
./run_coffee_one_traj_full.sh log
```

请求停止：

```bash
./run_coffee_one_traj_full.sh stop
```

停止请求会在当前 evaluate 小块完成后生效。再次执行 `start` 会从已落盘主键继续。自动化状态和日志分别保存在：

```text
/path/to/DCCP/dccp/outputs/state_selection_audit/coffee_one_traj_v1/automation_status.json
/path/to/DCCP/dccp/outputs/state_selection_audit/coffee_one_traj_v1/automation/run.log
```

## 分阶段运行

默认使用 5 轨迹 pilot。先跑 tiny 时覆盖配置：

```bash
cd /path/to/DCCP/dccp/experiments/state_selection_audit
export AUDIT_CONFIG="$PWD/configs/coffee_tiny.yaml"
```

1. 检查路径、GPU 和 LRM 服务状态：

```bash
./run_coffee_smoke.sh setup
```

2. 必须先通过状态恢复闸门：

```bash
./run_coffee_smoke.sh restore
```

它会从中间状态恢复两个全新环境，执行同一个动作块，比较两次恢复图像、下一图像、下一物理 state、成功标记和执行步数。采集 renderer 与首次恢复 renderer 的极少量初始化像素差只记录为诊断，不作为物理恢复失败。闸门失败时不要继续实验。

3. 收集冻结 VLA 的 nominal 轨迹：

```bash
./run_coffee_smoke.sh collect
```

`collect` 现在会硬性检查 `state_restore_gate.json` 的 `passed=true`；没有真正通过恢复闸门时不能绕过。每个阶段成功后都会自动刷新检查页，也可以随时手动生成：

```bash
./run_coffee_smoke.sh inspect
```

用 VS Code 打开：

```text
/path/to/DCCP/dccp/outputs/state_selection_audit/<run_name>/inspection/index.html
```

页面中的 nominal contact sheet 会按 VLA 决策时刻展示帧、决策编号和动作熵。确认轨迹画面、帧序和任务状态合理后，再启动 LRM。

4. 在 `vlm_reward` 环境、另一台机器或另一张空闲 GPU 上启动 Progress LRM：

```bash
conda activate vlm_reward
./start_progress_server.sh
curl --noproxy '*' http://127.0.0.1:8002/health
```

若服务在 neu_lab4，沿用项目已有 SSH 端口转发，让本机 `127.0.0.1:8002` 可达。这里只需要 progress 服务。

5. 冻结四种状态选择：

```bash
./run_coffee_smoke.sh select
```

选择阶段若发现已有 branch 结果，会拒绝重算。`SELECTIONS_FROZEN.json` 是防信息泄漏标记。完成后可以停止本机 LRM，释放 GPU。

再次运行 `inspect`。页面会增加 progress、曲率、动作熵、joint score 曲线，以及四种方法所选时刻的彩色竖线。此时先确认 eligible 状态和选择位置，再做 simulator branch。

6. 先只运行一个分支并保存视频：

```bash
./run_coffee_smoke.sh evaluate \
  --trajectory-id traj_0000 \
  --decision-index 2 \
  --candidate-index 0 \
  --evaluation-seed 101 \
  --max-branches 1 \
  --save-video
```

`decision-index` 必须从检查页显示的 eligible 状态中选择。然后：

```bash
./run_coffee_smoke.sh inspect
```

在页面直接播放该分支 MP4，检查恢复起点、首动作、后续 VLA 控制和任务终止是否合理。已完成但当时没有录视频的分支，可加 `--rerun-completed --save-video` 重跑并记录。

确认单分支后，可逐状态运行，例如限制一个状态的全部候选和 seed：

```bash
./run_coffee_smoke.sh evaluate \
  --trajectory-id traj_0000 \
  --decision-index 2 \
  --save-video
```

正式遍历全部有效状态时，为控制磁盘占用，默认不录视频：

```bash
./run_coffee_smoke.sh evaluate
```

该阶段支持断点续跑。每个候选在全新环境中恢复；同一状态/评估 seed 的所有候选使用相同的后续策略随机数序列，并共享相同的剩余全局步预算。

7. 完整性校验并生成表格：

```bash
./run_coffee_smoke.sh summarize
```

任何缺失或失败 branch 都会使汇总拒绝运行，避免用不完整数据得到偏置结论。

最后再运行一次 `inspect`，页面会显示最终方法主表。`inspect` 不使用 GPU，可在任何阶段重复执行。

所有 GPU 阶段都会在加载模型前打印 `nvidia-smi` 状态和计算进程。GPU 被占用时默认打印 `SKIPPED` 并安全退出，不抢占也不杀进程。

## 产物

```text
/path/to/DCCP/dccp/outputs/state_selection_audit/<run_name>/
├── manifest.json
├── nominal/                 # 帧、model XML、MuJoCo state、原动作和 token
├── selections/              # progress/曲率/熵、LRM cache、冻结选择与有效域
├── branches/                # 去重候选、逐 seed 回报和可选 MP4
├── reports/                 # 状态指标、逐轨迹表和总表
└── inspection/              # 离线 HTML、contact sheet 和选择时间轴 SVG
```

关键报告是 `reports/summary.csv`。改善量按 `best candidate success rate - nominal success rate` 计算；失败转成功率使用同 seed 的配对结果；oracle 与 random 使用相同有效域、预算和 NMS。oracle 为零时捕获比例记为 `null`。

入口使用 `umask 002`，并把专用输出根目录设为 `neu_lab2` 组和 setgid；neu_lab2 与 neu_lab4 新建的 run 都会继承共同组。不要同时写同一个 run。

## `evaluate` 的人工检查参数

```text
--trajectory-id      只运行一条轨迹，例如 traj_0000
--decision-index     只运行一个 eligible 决策时刻
--candidate-index    只运行一个候选，0 是 nominal 原动作
--evaluation-seed    只运行一个评估 seed
--max-branches       本次最多新执行多少个 branch
--save-video         保存 MP4
--video-fps          MP4 帧率，默认 20
--rerun-completed    允许重跑已经成功记录的主键，常用于补视频
```

过滤条件不匹配任何 eligible 状态时会直接报错，不会悄悄跑其他状态。每个 branch 开始和结束时都会在终端打印完整主键、状态、成功结果和视频路径。

## 测试

```bash
/path/to/conda_envs/dccp_env/bin/python -m pytest -q \
  /path/to/DCCP/dccp/experiments/state_selection_audit/tests

/path/to/conda_envs/dccp_env/bin/python -m compileall -q \
  /path/to/DCCP/dccp/experiments/state_selection_audit
```

更严格的约束见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。
