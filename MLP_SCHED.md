# Volc MLP 三队列动态任务分配器 (`mlp_sched.py`)

`scripts/mlp_sched.py` 是一个**零副作用、只读决策型**的多队列动态任务分配顾问工具。它针对火山引擎 MLP 的三类资源池：
1. **`dev`（本开发机队列）**：2 卡或 4 卡独占，零排队、零抢占；
2. **`proj`（项目非抢占池，`Preemptible: false`）**：48 卡共享池，运行无中断，但存在排队拥堵与大卡碎片风险；
3. **`pre`（大卡池抢占队列，`Preemptible: true`）**：容量充裕、排队快，但存在 `InstanceReclaimed` 抢占中断与回滚重排损耗；

求解黑盒优化映射 **$M = f(g, s, \text{DAG}; \beta)$**，输出每个未启动任务 $i$ 的目标队列分配行向量 $M_i \in \{0,1\}^4$（对应 `[none, dev, proj, pre]`）。

---

## 1. 核心原理与数据流

- **专有遥测数据库（`data/telemetry.db`）**：
  - 自动分页拉取 `volc ml_task list --showAllTasks` 中指定队列的历史与实时任务记录，结合本地 `nvidia-smi` 空闲卡检测。
  - 按卡数 $g \in \{1, 2, 4, 8\}$ 自动统计拟合：
    - `proj` 池在已等待 $t_i$ 秒条件下的贝叶斯后验剩余等待代价 $W_{\text{proj}}(g_i, t_i; \beta)$（自动感知当前是否有同规格任务卡在 `Queue` 状态）；
    - `pre` 池的排队代价 $W_{\text{pre}}(g_i, t_i; \beta)$、右删失泊松抢占率 $\hat{\lambda}_{\text{pre}}(g_i)$ 及期望中断回滚损耗 $P_{\text{pre}}(g_i; \beta)$。
- **输入规格 $(g_i, s_i) + \text{DAG}$**：
  - 仅针对**未开始运行（Unstarted）的任务 $i \in \mathcal{I}$**；
  - $g_i \in \{1, 2, 4, 8\}$：任务所需 GPU 卡数；
  - $s_i = (q_i, t_i)$：任务当前所在队列 $q_i \in \{\text{none}, \text{dev}, \text{proj}, \text{pre}\}$ 及已排队时长 $t_i$（未提交任务天然为 `("none", 0)`）；
  - $\text{DAG}$：任务间前置依赖关系（未完成依赖的任务固定输出 `M=[1,0,0,0]` 即 `HOLD (none)`）。
- **在线自动匹配（复用 `my_tasks.sh` 逻辑）**：
  - 你只需维护一份静态的 `tasks.yaml`（填写 `name`, `gpus`, `deps`）。
  - 每次运行 `mlp_sched.py` 时，脚本会自动在 `volc` 上按 `JobName` 查找你名下的任务：
    - 尚未提交 $\implies$ 自动设为 `s_i = (none, 0m)`；
    - 正在 `Queue` 排队 $\implies$ 自动填入当前队列（`proj`/`pre`）与真实已等待时长 $t_i = \text{now} - \text{CreateTime}$；
    - 已处于 `Running` $\implies$ 自动从待分配集中排除；
    - 已 `Success` 完成 $\implies$ 自动标记完成并解锁下游 `deps` 任务。

---

## 2. 命令速查

```bash
# 1. 同步 volc 队列状态更新 SQLite 数据库
./scripts/mlp_sched.py sync

# 2. 查看数据库当前拟合出的各卡数 (1/2/4/8卡) 排队分布与抢占 MTBF 参数
./scripts/mlp_sched.py stats
./scripts/mlp_sched.py stats --json

# 3. 基于任务清单文件求解分配矩阵 M = f(g, s)
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml

# 4. 覆盖风险厌恶系数 --beta (0=激进偏pre, 1=保守求稳) 或预留开发机卡数
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml --beta 0.6 --dev-reserved-gpus 1

# 5. 免配置文件模式：自动扫描你在 volc 上所有排队中的任务 + 命令行临时传入未提交任务卡数
./scripts/mlp_sched.py plan --pending-gpus 2,4,4,8

# 6. 持续监控刷新模式（只读不自动执行 submit/cancel，每 60s 刷新一次状态与建议矩阵 M）
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml --watch --interval 60

# 7. 输出 JSON 结构化矩阵（供自动化脚本或 Agent 解析）
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml --output json

# 8. 查看决策审计日志与 Agent 采纳/预测准确率统计
./scripts/mlp_sched.py audit
./scripts/mlp_sched.py audit --limit 30
./scripts/mlp_sched.py audit --json
```

---

## 3. 任务清单格式（`scripts/tasks.example.yaml`）

```yaml
config:
  beta: 0.3                  # 风险厌恶系数 in [0, 1]，可被 CLI --beta 覆盖
  dev_reserved_gpus: 0       # 本开发机预留给交互调试的卡数（默认 0）

tasks:
  # 平时只需填 name, gpus, deps（状态由脚本去 volc 自动匹配，未提交默认 queue="none", waited_min=0）
  - name: "sft-stage1-2gpu"
    gpus: 2
    deps: []

  - name: "rl-exp-a-4gpu"
    gpus: 4
    deps: []

  - name: "downstream-8gpu"
    gpus: 8
    deps: ["sft-stage1-2gpu"]

  # 如需离线模拟测试，也可显式指定 queue 和 waited_min 强制覆盖状态：
  - name: "stuck-proj-4gpu"
    gpus: 4
    queue: "proj"
    waited_min: 95
    deps: []
```

---

## 4. 输出动作解读

| 输出列 `M=[none,dev,proj,pre]` | `Action` 标识 | 含义与操作建议 |
| :--- | :--- | :--- |
| `[1, 0, 0, 0]` | `HOLD (none)` | 前置依赖未完成，暂不提交。 |
| `[0, 1, 0, 0]` | `none -> dev` / `proj -> dev` | 本开发机有足够空闲卡，建议直接在本地运行（若已在远端排队可 `cancel` 后拉回本地）。 |
| `[0, 0, 1, 0]` | `none -> proj` / `KEEP (proj)` | 48 卡非抢占池当前排队预期较短，建议以 `Preemptible: false` 提交或继续留在 `proj` 等待。 |
| `[0, 0, 0, 1]` | `none -> pre` / `proj -> pre` | `proj` 池拥堵或已排队超时（$t_i$ 过大触发重尾后验），建议以 `Preemptible: true` 提交/改交至抢占池。 |

---

## 5. 决策审计与闭环遥测 (`plan_audit_history`)

系统在 `data/telemetry.db` 中维护了 `plan_audit_history` 表，实现无须人工/Agent介入的**被动式闭环对账（Passive Ground-Truth Reconciliation）**：

1. **自动记录决策快照**：
   每次调用 `plan` 产生建议时，自动记下每个任务的推荐队列（`recommended_queue`）、模型预测的排队时间（`est_cost_min`、`proj_cost_min`、`pre_cost_min`）与参数 $\beta$。
2. **自动对账真实结果**：
   在后续执行 `sync` 时，系统扫描 `volc` 上由你提交的新任务：
   - 根据任务的 `Preemptible` 字段自动比对 Agent 最终实际提交的队列（`actual_queue`）；
   - 自动判定 Agent **是否遵从了建议（`adopted = 1/0`）**；
   - 在任务运行或结束时，自动记录真实排队时长（`actual_wait_min`）、实际运行小时（`actual_run_hrs`）以及是否被抢占（`was_preempted`）。
3. **闭环复盘统计（`audit` 命令）**：
   运行 `./scripts/mlp_sched.py audit` 会自动输出：
   - **Agent 采纳率（Adoption Rate %）**：监督 Agent 操作是否遵从最优队列建议；
   - **排队时间预估误差（MAE）**：衡量模型预测排队时间的准确度，用于持续校准 $\mu_L, \mu_H$ 与 $\kappa_{\text{cong}}$；
   - **抢占发生率**：统计被派往抢占池的任务实际被正主回收的频率，用于校准 $\lambda_{\text{pre}}$。
