# Volc-MLP: Intelligent Multi-Queue Dynamic Scheduler & Agent Skill

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Volcengine MLP](https://img.shields.io/badge/Volcengine-MLP-orange.svg)](https://www.volcengine.com/product/mlp)

**Volc-MLP** is a production-tested multi-queue dynamic scheduling optimizer, management toolchain, and AI Agent Skill designed for **Volcengine Machine Learning Platform (火山引擎机器学习平台 MLP)**.

It solves the practical resource allocation and preemption-tradeoff problem when training deep learning models across heterogeneous resource pools, providing **read-only advisory planning**, **telemetry tracking**, and **closed-loop decision audits**.

---

## The Problem: The Three-Queue Dilemma

When executing training jobs on Volcengine MLP, engineers typically juggle three distinct resource queues:

| Queue | Capacity | Wait Latency | Preemption Risk | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **1. DevInstance (`dev`)** | 1 ~ 4 dedicated GPUs | **Zero wait** | **Zero preemption** | Interactive debug, small tests, critical blocking nodes |
| **2. Project Pool (`proj`)** | Shared quota (e.g., 48 GPUs) | Heavy queuing / fragmentation | **Zero preemption** (`Preemptible: false`) | Medium/long tasks, jobs that cannot afford checkpoint rollbacks |
| **3. Preemptible Pool (`pre`)** | Borrowable idle cluster capacity | **Fast queueing** | **High preemption risk** (`InstanceReclaimed`) | Parallel sweeps, hyperparameter tuning, dense checkpointing tasks |

Without systematic modeling, users frequently face two failure modes:
1. **Queuing Paralysis**: Submitting all tasks to `proj` and waiting hours behind team members;
2. **Preemption Churn**: Blindly submitting large 8-GPU jobs to `pre`, getting evicted repeatedly, and losing progress.

---

## Key Features

- **Exact Quadratic 0-1 Allocation ($M = f(g, s, \text{DAG}; \beta)$)**:
  Dynamically maps each unstarted task to `[none, dev, proj, pre]` via exact Dynamic Programming, factoring in GPU count $g_i$, elapsed wait time $t_i$, DAG dependencies, and self-congestion.
- **Bayesian Regime-Switching Wait Estimator**:
  Distinguishes between "Light" ($L$) and "Locked/Heavy" ($H$) queue regimes. Automatically detects when colleagues' long-running jobs lock the pool and computes risk-adjusted residual waiting times.
- **Right-Censored Poisson Preemption Modeling**:
  Learns real preemption rates $\hat{\lambda}_{\text{pre}}(g)$ from cluster history, properly treating running and completed tasks as censored survival intervals.
- **Closed-Loop Decision Audit (`audit`)**:
  Passively tracks whether human engineers or AI Agents followed the recommendation (`adopted = 1/0`), measures prediction error (MAE), and audits counterfactual regret.
- **Zero-Side-Effect Advisory Mode**:
  Operates purely as a read-only advisor—never mutates, deletes, or submits tasks without explicit user intent.
- **Ready-to-Use Agent Skill**:
  Standardized `SKILL.md` for seamless plug-and-play with **Google Antigravity**, **Claude Code**, and **Codex CLI**.

---

## Quickstart

### 1. Installation

Clone the repository and install the lightweight dependency (`pyyaml`):

```bash
git clone https://github.com/SunnyYYLin/volc-mlp.git
cd volc-mlp
pip install -r requirements.txt  # or: pip install -e .
```

### 2. Configuration

Copy `config.example.yaml` to `config.local.yaml` (which is git-ignored):

```bash
cp config.example.yaml config.local.yaml
```

Fill in your cluster identifiers:
```yaml
queue_id: "q-your-resource-queue-id"
user_id: "your-iam-subuser-id"
dev_reserved_gpus: 0
beta: 0.3
```

*(Alternatively, export environment variables: `export VOLC_QUEUE_ID="q-xxx"` and `export VOLC_USER_ID="xxx"`).*

### 3. Usage

#### A. View Learned Cluster Queue Priors
```bash
./scripts/mlp_sched.py stats
```
```text
Local DevInstance GPUs : 2 free / 2 total
--------------------------------------------------------------------------------------------------------
GPUs   Proj Light(m)   Proj Heavy(m)   Proj P(Light)   Proj Queued   Pre Wait(m)   Pre MTBF(h)   Samples(P/E)
--------------------------------------------------------------------------------------------------------
1      3.0             48.0            0.65            0             1.9           50.44         0/2
2      5.0             72.0            0.55            0             3.0           5.56          0/0
4      5.0             267.6           0.53            0             5.0           3.33          23/0
8      14.3            297.5           0.37            0             9.0           1.82          2/0
```

#### B. Solve Allocation Plan for Unstarted Tasks
Input task lists via YAML or ad-hoc GPU counts:

```bash
# 1. From YAML definition with DAG dependencies:
./scripts/mlp_sched.py plan --tasks scripts/tasks.example.yaml --beta 0.3

# 2. Or pass ad-hoc unsubmitted tasks directly:
./scripts/mlp_sched.py plan --pending-gpus 2,4,4,8
```

Output:
```text
====================================================================================================================
[2026-11-09 20:00:00 UTC] Volc MLP Dynamic Allocation Plan  |  beta=0.30  |  DevInstance Free GPUs: 2/2
====================================================================================================================
Task ID                  g_i  State s_i=(q_i, t_i) M=[none,dev,proj,pre]   Action           Reason
--------------------------------------------------------------------------------------------------------------------
sft-stage1-2gpu          2    (none, 0.0m)         [0,   1,   0,    0  ]   none -> dev      DevInstance has 2 free GPUs (0 wait, 0 preemption)
rl-exp-a-4gpu            4    (none, 0.0m)         [0,   0,   0,    1  ]   none -> pre      Submit preemptible (pre cost ~43.6m vs proj ~340.5m + congestion)
rl-exp-b-4gpu            4    (none, 0.0m)         [0,   0,   0,    1  ]   none -> pre      Submit preemptible (pre cost ~43.6m vs proj ~340.5m + congestion)
stuck-proj-4gpu          4    (proj, 95.0m)        [0,   0,   0,    1  ]   proj -> pre      Proj queue timeout after 95.0m (proj rem cost ~340.7m > pre cost ~45.6m)
downstream-8gpu          8    (none, 0.0m)         [1,   0,   0,    0  ]   HOLD (none)      Waiting on DAG deps: sft-stage1-2gpu
--------------------------------------------------------------------------------------------------------------------
```

#### C. Inspect Decision Audit Log & Agent Compliance
```bash
./scripts/mlp_sched.py audit
```
```text
======================================================================================================================
Volc MLP Decision Audit Log & Telemetry
======================================================================================================================
Audit ID   Timestamp           Task / Job             GPUs  Rec    Actual  Adopted   Est(m)   Act(m)   Preempt  Status
----------------------------------------------------------------------------------------------------------------------
#1         2026-11-09 19:28:51 pending_1_2gpu         2     pre    pre     YES       24.6     21.2     NO       Running
#2         2026-11-09 19:28:51 pending_2_4gpu         4     pre    pre     YES       43.6     40.1     NO       Success
#3         2026-11-09 19:28:51 pending_3_8gpu         8     pre    proj    NO        91.7     280.4    NO       Killed
----------------------------------------------------------------------------------------------------------------------
Summary Stats: Total Tracked=3 | Reconciled=3 | Agent Adoption Rate=66.7% (2/3) | Wait MAE=3.5m | Preemptions in Pre=0/2
```

---

## Utility Tools

Located in `tools/`:
- `tools/my_tasks.sh`: Fast paginated search for all active & finished tasks under your IAM user account.
- `tools/task_logs.sh <task_id> [lines] [keyword]`: Safe wrapper for streaming task container logs with automatic pod instance resolution.
- `tools/task_reminder.sh <job_id> [timeout] [interval]`: Non-blocking state-change reminder script for long-running workflows.

---

## Repository Structure

```text
volc-mlp/
├── README.md                 # Project overview and user guide
├── LICENSE                   # MIT License
├── .gitignore                # Security filter (databases, local credentials)
├── pyproject.toml            # Project packaging metadata
├── config.example.yaml       # Configuration template
├── SKILL.md                  # Custom skill interface for AI Coding Agents
├── MLP_SCHED.md              # In-depth mathematical formulation & telemetry docs
├── references/
│   ├── doc.md                # Official Volcengine CLI manual
│   └── example.yaml          # Task submission YAML specification
├── scripts/
│   ├── mlp_sched.py          # Main optimization & telemetry engine
│   └── tasks.example.yaml    # Workload input schema example
└── tools/
    ├── my_tasks.sh           # Account task listing utility
    ├── task_logs.sh          # Container log streaming utility
    └── task_reminder.sh      # Background task monitor
```

---

## Integration with AI Agents

To use `volc-mlp` as an Agent Skill in **Antigravity**, **Claude Code**, or **Codex**:
1. Place this directory under `~/.agents/skills/volc-mlp` or your workspace skills directory.
2. The agent automatically discovers `SKILL.md` and uses the toolchain to optimize your queue submissions and monitor remote jobs.

---

## License

This project is licensed under the [MIT License](LICENSE).
