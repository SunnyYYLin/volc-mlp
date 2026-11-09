---
name: volc-mlp
description: 使用火山引擎 `volc` CLI 管理机器学习平台的训练任务、日志、开发机和模型。用于提交、查询、监控、停止 MLP 任务，查看开发机，或管理模型；只使用稳定的 `volc ml_task`、`volc ml_devinstance` 和 `volc ml_model` 命令。
---

# Volc MLP 使用指南

## 账户与环境配置

在根目录配置 `config.local.yaml`（参考 `config.example.yaml`）或设置环境变量：
- 资源队列：`VOLC_QUEUE_ID` 或 `config.local.yaml` 中的 `queue_id`
- IAM 子用户：`VOLC_USER_ID` 或 `config.local.yaml` 中的 `user_id`

## 命令约束

只使用 `volc` CLI。不要使用另一套实验性客户端。

`volc` 可能输出 metrics 警告或升级提示，只要退出码为 0 即有效；解析 JSON 时提取首个 `[` 与末尾 `]` 之间的内容。

## 命令速查

```bash
# 提交任务（返回 JSON 含 Id）
volc ml_task submit --conf task.yaml --output json

# 查询任务（支持 --output json，全状态覆盖）
volc ml_task get --id <task_id> --output json

# 列表查询（支持 --output json，默认只显示非终态）
volc ml_task list --status Running,Queue,Staging,Killing --limit 100 --output json
# 查看终态任务需显式指定
volc ml_task list --status Failed,Success,Killed --limit 20 --output json
# 按名称/ID 模糊搜索
volc ml_task list --name <keyword> --status Running,Failed,Success --output json

# 实例列表（--id 或 -i 接 task id）
volc ml_task instance list -i <task_id>

# 日志（--task/-t 和 --instance/-i 都是必填）
volc ml_task logs -t <task_id> -i worker_0 -l 500
# 从启动时间正序看（-r）
volc ml_task logs -t <task_id> -i worker_0 -l 500 -r
# 按内容过滤（-c）
volc ml_task logs -t <task_id> -i worker_0 -l 500 -c "Error"
# 实时跟踪
volc ml_task logs -t <task_id> -i worker_0 -f --timestamp

# 停止任务
volc ml_task cancel --id <task_id>

# 开发机与模型（默认进入 TUI，加 `echo q |` 可免挂起直接打印）
echo q | volc ml_devinstance list --queueIds <queue_id> --limit 100
echo q | volc ml_devinstance get --id <devinstance_id>
echo q | volc ml_model list --limit 100
```

### 注意事项

- `volc ml_task get --id <id> --output json` 可直接获取指定任务状态（含终态），比 `list --name` 更稳定精准。
- `volc ml_task logs` 的 `--task` 和 `--instance` 都是必填参数。instance id 通常是 `worker_0`（多节点为 `worker_1`...）；注意 `instance list` 输出的 PodName（如 `t-xxx-worker-0`）不能直接传给 `logs -i`。
- `volc ml_task instance list` 的参数是 `--id/-i`（接 task id），不是 `--task`。
- `volc ml_task list` 默认只显示非终态任务；且若不带 `--output json` 也会进入交互式 TUI。
- 开发机与模型命令不支持 `--output json`，在脚本或自动化中必须前置 `echo q |` 以防卡死。

## 提交流程

1. 从 `references/example.yaml` 复制任务配置并按任务修改。
2. 检查 `ResourceQueueID`、GPU Flavor、挂载路径、入口命令和私密环境变量。
3. 抢占式任务必须配置 `RetryOptions`，默认只对 `InstanceReclaimed` 重试；不要默认对代码失败无限重试。
4. 使用 `volc ml_task submit --conf <yaml> --output json` 提交。
5. 从返回 JSON 记录任务 ID，并使用 `volc ml_task get --id <id> --output json` 验证状态。
6. 在当前临时目录维护任务 ID 和状态列表。不要依赖无过滤的全队列列表。

## 任务配置

```yaml
TaskName: "example"
Entrypoint: "nvidia-smi"
ImageUrl: "vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.8.1-py3.12-ubuntu22.04"
ResourceQueueID: "q-your-resource-queue-id"
Framework: "Custom"
TaskRoleSpecs:
    - RoleName: "worker"
      RoleReplicas: 1
      Flavor: "ml.pni2.3xlarge"
Description: "任务说明"
AccessType: "Private"
Priority: 6
Preemptible: true
RetryOptions:
    EnableRetry: true
    MaxRetryTimes: 64
    IntervalSeconds: 60
    PolicySets:
        - InstanceReclaimed
Storages:
    - Type: Vepfs
      MountPath: /vepfs/your-project/your-username
      SubPath: your-project/your-username
    - Type: TosFuse
      MountPath: /tos/your-tos-bucket
      Bucket: your-tos-bucket
      Prefix: /
Envs:
    - Name: "VEPFS"
      Value: "/vepfs/your-project/your-username"
```

GPU Flavor：`3xlarge=1GPU`、`7xlarge=2GPU`、`14xlarge=4GPU`、`28xlarge=8GPU`。

### 常用镜像推荐

- **基础 CUDA 镜像**：`vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.8.1-py3.12-ubuntu22.04`
- **Isaac 任务镜像**：当需要使用 Isaac（Isaac Sim / Isaac Lab）时，使用以下镜像：
  `vemlp-cn-beijing.cr.volces.com/preset-images/physical-ai:isaacsim-v5.1.0_lab-v2.3.2`

## 工具脚本

```bash
./tools/my_tasks.sh
./tools/my_tasks.sh --state Running
./tools/task_logs.sh <task_id> [lines] [keyword]
./tools/task_reminder.sh <task_id> 300 10
```

## 三队列动态任务分配器 (`mlp_sched.py`)

当有多个不同卡数的待跑/排队任务需要在**本开发机 (`dev`)**、**48 卡非抢占项目池 (`proj`)** 与**抢占式大卡池 (`pre`)** 之间决策分配或判断何时从 `proj` 转投 `pre` 时，使用 `./scripts/mlp_sched.py`（自带 SQLite 遥测数据库统计各卡数排队时长与抢占率，求解 $M = f(g, s, \text{DAG}; \beta)$）：

```bash
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml [--beta 0.3] [--watch]
./scripts/mlp_sched.py audit   # 查看历史决策、Agent 采纳率与时间预估误差
```

详细建模参数、YAML 格式与子命令说明见 `MLP_SCHED.md`。

## 常见问题

- `当前队列剩余配额不足`：改用 `Preemptible: true` 并配置 `InstanceReclaimed` 重试。
- 持续排队且提示 GPU 碎片：继续监控，或降低 Flavor；不要重复提交同一任务。
- `invalid credential`：检查 `~/.volc/config` 和 `~/.volc/credentials`，不要打印其中内容。
- 创建任务后立即失败：先用 `volc ml_task logs -t <id> -i worker_0 -l 500` 获取日志，再决定是否重试。
- 日志看不到根因：用 `-r` 从启动正序看，或用 `-c "Error"` 过滤。Worker 子进程崩溃时父进程只打 "initialization failed"，需要看更早的日志。

详细 CLI 参考见 `references/doc.md`。
