#!/usr/bin/env python3
"""
Volc MLP Dynamic Task Allocator & Telemetry Advisor (mlp_sched.py)

Solves the practical multi-queue allocation problem:
    M = f(g, s, DAG; beta)
where:
  - Input per unstarted task i:
      g_i in {1, 2, 4, 8, ...} (GPU count)
      s_i = (q_i, t_i)         (current queue in {none, dev, proj, pre} and waited time t_i >= 0)
      deps                     (prerequisite task IDs in DAG)
  - Output:
      Assignment matrix M_{i,j} in {0, 1} for j in {none, dev, proj, pre},
      recommending which queue each unstarted task should be placed/moved into.

Read-only advisor: syncs telemetry from `volc ml_task list` and `nvidia-smi` into SQLite
(`data/telemetry.db`) and outputs optimal allocation plans without mutating live jobs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # Fallback to JSON if PyYAML is unavailable

QueueName = Literal["none", "dev", "proj", "pre"]
QUEUE_COLS: Tuple[QueueName, ...] = ("none", "dev", "proj", "pre")

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = SKILL_DIR / "data" / "telemetry.db"


def load_local_config() -> Dict[str, Any]:
    """Load default configurations from config.local.yaml or environment variables."""
    cfg: Dict[str, Any] = {}
    local_cfg_path = SKILL_DIR / "config.local.yaml"
    if local_cfg_path.exists() and yaml is not None:
        try:
            with open(local_cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
        except Exception:
            pass
    if "VOLC_QUEUE_ID" in os.environ:
        cfg["queue_id"] = os.environ["VOLC_QUEUE_ID"]
    if "VOLC_USER_ID" in os.environ:
        cfg["user_id"] = os.environ["VOLC_USER_ID"]
    return cfg


_LOCAL_CFG = load_local_config()
DEFAULT_QUEUE_ID = str(_LOCAL_CFG.get("queue_id") or os.environ.get("VOLC_QUEUE_ID") or "")
DEFAULT_USER_ID = str(_LOCAL_CFG.get("user_id") or os.environ.get("VOLC_USER_ID") or "")

FLAVOR_GPU_MAP: Dict[str, int] = {
    "ml.pni2.3xlarge": 1,
    "ml.pni2.7xlarge": 2,
    "ml.pni2.14xlarge": 4,
    "ml.pni2.28xlarge": 8,
}


# ============================================================================
# 1. Core Data Structures & Interfaces
# ============================================================================

@dataclass
class TaskState:
    """Minimal input state (g_i, s_i) + DAG edges for an unstarted task i."""
    task_id: str
    gpus: int                        # g_i in {1, 2, 4, 8}
    current_queue: QueueName = "none"# q_i in {'none', 'dev', 'proj', 'pre'}
    waited_sec: float = 0.0          # t_i >= 0 (seconds already waited in q_i)
    deps: List[str] = field(default_factory=list)
    job_id: Optional[str] = None     # Matched online volc JobId if currently queued
    dag_descendants: int = 0         # Number of downstream tasks blocked by this task


@dataclass
class QueuePriors:
    """Estimated statistical parameters for a specific GPU count g from SQLite."""
    gpus: int
    # Project non-preemptible pool (48 GPUs)
    proj_mu_light_sec: float         # Mean wait time in light regime L (sec)
    proj_mu_heavy_sec: float         # Mean wait time in heavy/locked regime H (sec)
    proj_p0_light: float             # Empirical probability of regime L
    proj_active_queued_count: int    # Real-time number of tasks of size <= g queuing in proj
    proj_max_queued_wait_sec: float  # Real-time longest wait among currently queued proj tasks
    proj_recent_launch_wait_sec: float # Mean wait of tasks launched in the last 3 hours in proj
    proj_visible_running_gpus: int   # Currently visible Running non-preemptible GPUs in queue
    # Preemptible large pool
    pre_mu_wait_sec: float           # Mean initial wait time in pre (sec)
    pre_std_wait_sec: float          # Std of wait time in pre (sec)
    pre_lambda_per_hr: float         # Preemption rate (interruptions per hour)
    pre_requeue_wait_sec: float      # Mean high-priority requeue wait after preemption (sec)
    avg_run_hrs: float               # Reference runtime horizon (hours)
    sample_count_proj: int = 0
    sample_count_pre: int = 0

    def proj_residual_wait_cost(self, waited_sec: float, beta: float) -> float:
        """
        Compute risk-adjusted expected residual wait time in `proj`:
            W_proj(g, t; beta) = E[W - t | W > t] + z(beta) * Std[W - t | W > t]
        Combines:
          1. Historical two-regime (Light mu_L / Heavy mu_H) parameters
          2. Real-time queued tasks (`proj_active_queued_count`, `proj_max_queued_wait_sec`)
          3. Recent launch wait times in the last 3 hours (`proj_recent_launch_wait_sec`)
             and visible running GPUs (`proj_visible_running_gpus`).
        """
        mu_L = max(60.0, self.proj_mu_light_sec)
        mu_H = max(mu_L * 3.0, self.proj_mu_heavy_sec)

        # Base p0 from historical frequency, modulated by live queue & recent 3h congestion
        p0 = self.proj_p0_light
        if self.proj_active_queued_count == 0 and self.proj_recent_launch_wait_sec < mu_L * 1.5:
            # Pool is genuinely quiet right now
            if self.proj_visible_running_gpus + self.gpus <= 28:
                p0 = max(p0, 0.80)
            else:
                p0 = max(p0, 0.60)
            effective_obs_wait = max(0.0, waited_sec)
        else:
            # Either tasks are currently stuck in Queue, or recent launches in the last 3h waited a long time
            recent_signal = 0.6 * self.proj_recent_launch_wait_sec if waited_sec <= 0.0 else 0.0
            effective_obs_wait = max(0.0, waited_sec, self.proj_max_queued_wait_sec, recent_signal)

        # Bayesian posterior probability of being in Heavy state H given wait > effective_obs_wait
        surv_L = p0 * math.exp(-effective_obs_wait / mu_L)
        surv_H = (1.0 - p0) * math.exp(-effective_obs_wait / mu_H)
        denom = surv_L + surv_H
        p_H_post = (surv_H / denom) if denom > 1e-12 else 1.0

        # Conditional residual mean & standard deviation
        mean_rem = (1.0 - p_H_post) * mu_L + p_H_post * mu_H
        std_rem = (1.0 - p_H_post) * (0.5 * mu_L) + p_H_post * (0.65 * mu_H)

        # Backlog offset if newly joining behind existing visibly queued tasks
        backlog_sec = 0.0
        if waited_sec <= 0.0 and self.proj_active_queued_count > 0:
            backlog_sec = min(mu_H, 0.35 * self.proj_active_queued_count * mu_L)

        z_beta = 1.4 * max(0.0, min(1.0, beta))
        return backlog_sec + mean_rem + z_beta * std_rem

    def pre_residual_wait_cost(self, waited_sec: float, beta: float) -> float:
        """Compute risk-adjusted residual wait time in `pre`."""
        mu = max(30.0, self.pre_mu_wait_sec)
        inflation = 1.0 + 0.25 * math.tanh(max(0.0, waited_sec - mu) / max(300.0, mu))
        mean_rem = mu * inflation
        std_rem = max(30.0, 0.5 * self.pre_std_wait_sec * inflation)
        z_beta = 1.4 * max(0.0, min(1.0, beta))
        return mean_rem + z_beta * std_rem

    def pre_preemption_penalty_sec(self, beta: float, dag_descendants: int = 0) -> float:
        """
        Compute equivalent interruption & rollback penalty P_pre(g; beta) in seconds.
        Accounts for expected number of preemptions over typical runtime, lost rollback progress,
        60s retry interval, high-priority requeue wait, container staging, and tail variance.
        """
        expected_interruptions = self.pre_lambda_per_hr * self.avg_run_hrs
        # Typical rollback loss (~15 min) + 60s cooldown + requeue wait + 180s staging
        per_interruption_loss_sec = 900.0 + 60.0 + self.pre_requeue_wait_sec + 180.0
        mean_pen_sec = expected_interruptions * per_interruption_loss_sec
        std_pen_sec = math.sqrt(max(0.1, expected_interruptions)) * per_interruption_loss_sec

        z_beta = 1.6 * max(0.0, min(1.0, beta))
        dag_weight = 1.0 + 0.35 * min(5, max(0, dag_descendants))
        return dag_weight * (mean_pen_sec + z_beta * std_pen_sec)


@dataclass
class AllocationDecision:
    """Single row of output decision matrix M along with human-readable explanation."""
    task_id: str
    gpus: int
    from_queue: QueueName
    to_queue: QueueName
    waited_min: float
    row_M: Dict[QueueName, int]
    cost_dev_min: Optional[float]
    cost_proj_min: Optional[float]
    cost_pre_min: Optional[float]
    action_label: str
    reason: str


@dataclass
class AllocationPlanResult:
    """Complete output of M = f(g, s, DAG; beta)."""
    timestamp: str
    beta: float
    dev_total_gpus: int
    dev_free_gpus: int
    active_running_tasks: List[str]
    completed_tasks: List[str]
    matrix_M: Dict[str, Dict[QueueName, int]]
    decisions: List[AllocationDecision]
    priors_by_gpu: Dict[int, QueuePriors]


# ============================================================================
# 2. Dedicated SQLite Telemetry Database (`data/telemetry.db`)
# ============================================================================

class TelemetryDB:
    """
    Manages the SQLite telemetry database for Volc MLP tasks and estimates
    queue wait distributions and preemption rates per GPU count g in {1, 2, 4, 8}.
    """

    # Sensible cold-start priors (used when historical sample count is small)
    DEFAULT_PRIORS: Dict[int, Dict[str, float]] = {
        1: {
            "proj_mu_L": 180.0,    # 3 min when free/light
            "proj_mu_H": 3600.0,   # 60 min when 48-GPU pool is full
            "proj_p0": 0.65,
            "pre_wait": 120.0,     # 2 min initial wait in preemptible pool
            "pre_lambda": 0.12,    # ~1 preemption per 8.3 hrs
            "pre_requeue": 90.0,   # 1.5 min high-priority requeue wait
            "avg_run_hrs": 3.0,
        },
        2: {
            "proj_mu_L": 300.0,    # 5 min
            "proj_mu_H": 5400.0,   # 90 min
            "proj_p0": 0.55,
            "pre_wait": 180.0,     # 3 min
            "pre_lambda": 0.18,    # ~1 preemption per 5.5 hrs
            "pre_requeue": 120.0,  # 2 min
            "avg_run_hrs": 3.5,
        },
        4: {
            "proj_mu_L": 480.0,    # 8 min
            "proj_mu_H": 9000.0,   # 150 min (2.5 hrs)
            "proj_p0": 0.45,
            "pre_wait": 300.0,     # 5 min
            "pre_lambda": 0.30,    # ~1 preemption per 3.3 hrs
            "pre_requeue": 180.0,  # 3 min
            "avg_run_hrs": 4.0,
        },
        8: {
            "proj_mu_L": 900.0,    # 15 min
            "proj_mu_H": 16200.0,  # 270 min (4.5 hrs, whole-node fragmentation)
            "proj_p0": 0.30,
            "pre_wait": 540.0,     # 9 min
            "pre_lambda": 0.55,    # ~1 preemption per 1.8 hrs
            "pre_requeue": 240.0,  # 4 min
            "avg_run_hrs": 5.0,
        },
    }

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_history (
                    job_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    queue_id TEXT NOT NULL,
                    preemptible INTEGER NOT NULL,
                    priority INTEGER DEFAULT 4,
                    gpus INTEGER NOT NULL,
                    flavor TEXT,
                    replicas INTEGER DEFAULT 1,
                    create_time TEXT NOT NULL,
                    launch_time TEXT,
                    finish_time TEXT,
                    status TEXT NOT NULL,
                    exit_code INTEGER DEFAULT 0,
                    is_requeue INTEGER DEFAULT 0,
                    wait_sec REAL NOT NULL,
                    wait_censored INTEGER NOT NULL,
                    run_sec REAL DEFAULT 0.0,
                    preempted INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_history_queue_gpu
                    ON task_history(queue_id, preemptible, gpus, status);
                CREATE INDEX IF NOT EXISTS idx_task_history_name
                    ON task_history(job_name, create_time);

                CREATE TABLE IF NOT EXISTS plan_audit_history (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    gpus INTEGER NOT NULL,
                    from_queue TEXT NOT NULL,
                    recommended_queue TEXT NOT NULL,
                    est_cost_min REAL NOT NULL,
                    proj_cost_min REAL,
                    pre_cost_min REAL,
                    beta REAL NOT NULL,
                    action_label TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    actual_job_id TEXT,
                    actual_queue TEXT,
                    adopted INTEGER,
                    actual_wait_min REAL,
                    actual_run_hrs REAL,
                    was_preempted INTEGER DEFAULT 0,
                    status TEXT,
                    reconciled INTEGER DEFAULT 0,
                    reconciled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_job_name ON plan_audit_history(job_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_reconciled ON plan_audit_history(reconciled);
                """
            )

    @staticmethod
    def parse_utc(ts_str: Optional[str]) -> Optional[dt.datetime]:
        if not ts_str:
            return None
        s = ts_str.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def parse_gpus_from_specs(specs: Optional[List[Dict[str, Any]]]) -> Tuple[int, str, int]:
        if not specs:
            return 1, "unknown", 1
        total_gpus = 0
        flavor_name = "unknown"
        total_replicas = 0
        for role in specs:
            replicas = int(role.get("RoleReplicas") or 1)
            flavor = (
                role.get("ResourceSpecId")
                or (role.get("ResourceSpec") or {}).get("FlavorID")
                or ""
            )
            flavor_name = flavor or flavor_name
            total_replicas += replicas
            gpus_per_rep = FLAVOR_GPU_MAP.get(flavor)
            if gpus_per_rep is None:
                # Fallback regex for *xlarge naming
                m = re.search(r"(\d+)xlarge", flavor)
                if m:
                    xl = int(m.group(1))
                    gpus_per_rep = max(1, round(xl / 3.5))
                else:
                    gpus_per_rep = 1
            total_gpus += gpus_per_rep * replicas
        return max(1, total_gpus), flavor_name, max(1, total_replicas)

    def needs_sync(self, ttl_sec: float = 90.0) -> bool:
        """Return True if DB is empty or the latest sync was more than `ttl_sec` seconds ago."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(updated_at) AS last_ts FROM task_history").fetchone()
        if not row or not row["last_ts"]:
            return True
        last_dt = self.parse_utc(row["last_ts"])
        if last_dt is None:
            return True
        age = (dt.datetime.now(dt.timezone.utc) - last_dt).total_seconds()
        return age > ttl_sec

    def sync_from_volc(
        self,
        queue_id: str = DEFAULT_QUEUE_ID,
        user_id: str = DEFAULT_USER_ID,
        active_pages: int = 8,
        terminal_pages: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Fetch active and recent terminal tasks from `volc ml_task list`, follow up on
        previously active tasks in `task_history`, and update the SQLite database.
        Returns the list of all currently active tasks in `queue_id` (plus any owned by `user_id`).
        """
        now_utc = dt.datetime.now(dt.timezone.utc)
        now_iso = now_utc.strftime("%Y-%m-%d %H:%M:%S")

        active_raw = self._fetch_volc_pages(
            statuses="Queue,Staging,Running,Killing,Initialized",
            max_pages=active_pages,
        )
        terminal_raw = self._fetch_volc_pages(
            statuses="Success,Failed,Killed",
            max_pages=terminal_pages,
        )

        # Filter to target queue OR tasks created by user_id
        relevant: Dict[str, Dict[str, Any]] = {}
        for item in active_raw + terminal_raw:
            jid = str(item.get("JobId") or "")
            if not jid:
                continue
            if item.get("ResourceQueueId") == queue_id or str(item.get("Creator")) == str(user_id):
                relevant[jid] = item

        # Follow up in parallel on any task in DB that was previously in non-terminal state
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM task_history WHERE status IN ('Queue', 'Staging', 'Running', 'Killing', 'Initialized')"
            ).fetchall()
        missing_active_ids = [r["job_id"] for r in rows if r["job_id"] not in relevant][:25]
        if missing_active_ids:
            with ThreadPoolExecutor(max_workers=min(6, len(missing_active_ids))) as pool:
                fut_map = {pool.submit(self._fetch_single_task, jid): jid for jid in missing_active_ids}
                for fut in as_completed(fut_map):
                    fetched = fut.result()
                    if fetched and fetched.get("JobId"):
                        relevant[str(fetched["JobId"])] = fetched

        # Upsert all relevant records into SQLite
        with self._connect() as conn:
            # Load existing job names to detect retry/requeue instances
            existing_names: Dict[str, List[Tuple[str, str]]] = {}
            for r in conn.execute("SELECT job_id, job_name, create_time FROM task_history"):
                existing_names.setdefault(r["job_name"], []).append((r["job_id"], r["create_time"]))

            for jid, item in relevant.items():
                jname = str(item.get("JobName") or "")
                creator = str(item.get("Creator") or "")
                qid = str(item.get("ResourceQueueId") or queue_id)
                preemptible = 1 if item.get("Preemptible") else 0
                priority = int(item.get("Priority") or 4)
                status = str(item.get("Status") or "Unknown")
                exit_code = int(item.get("ExitCode") or 0)
                gpus, flavor, replicas = self.parse_gpus_from_specs(item.get("TaskRoleSpecs"))

                ctime_str = str(item.get("CreateTime") or "")
                ltime_str = str(item.get("LaunchTime") or "")
                ftime_str = str(item.get("FinishTime") or "")

                ctime = self.parse_utc(ctime_str) or now_utc
                ltime = self.parse_utc(ltime_str)
                ftime = self.parse_utc(ftime_str)

                # Wait time calculation
                if ltime is not None:
                    wait_sec = max(0.0, (ltime - ctime).total_seconds())
                    wait_censored = 0
                else:
                    end_ref = ftime if ftime is not None else now_utc
                    wait_sec = max(0.0, (end_ref - ctime).total_seconds())
                    wait_censored = 1

                # Run duration calculation
                if ltime is not None:
                    end_ref = ftime if ftime is not None else now_utc
                    run_sec = max(0.0, (end_ref - ltime).total_seconds())
                else:
                    run_sec = 0.0

                # Detect if this is a requeue of an earlier preemptible job with the same name
                prior_jobs = [
                    ct for other_id, ct in existing_names.get(jname, [])
                    if other_id != jid and ct < ctime_str
                ]
                is_requeue = 1 if (preemptible == 1 and (len(prior_jobs) > 0 or "retry" in jname.lower())) else 0

                # Preemption detection: preemptible job that launched and ended in Killed / non-zero exit
                preempted = 1 if (
                    preemptible == 1
                    and ltime is not None
                    and status in ("Killed", "Failed")
                ) else 0

                conn.execute(
                    """
                    INSERT INTO task_history (
                        job_id, job_name, creator, queue_id, preemptible, priority,
                        gpus, flavor, replicas, create_time, launch_time, finish_time,
                        status, exit_code, is_requeue, wait_sec, wait_censored,
                        run_sec, preempted, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        launch_time=excluded.launch_time,
                        finish_time=excluded.finish_time,
                        status=excluded.status,
                        exit_code=excluded.exit_code,
                        is_requeue=excluded.is_requeue,
                        wait_sec=excluded.wait_sec,
                        wait_censored=excluded.wait_censored,
                        run_sec=excluded.run_sec,
                        preempted=excluded.preempted,
                        updated_at=excluded.updated_at
                    """,
                    (
                        jid, jname, creator, qid, preemptible, priority,
                        gpus, flavor, replicas, ctime_str, ltime_str or None, ftime_str or None,
                        status, exit_code, is_requeue, wait_sec, wait_censored,
                        run_sec, preempted, now_iso,
                    ),
                )

        # Auto-reconcile previous audits with updated task_history
        self.reconcile_audits(user_id=user_id)

        active_list = [
            item for item in relevant.values()
            if item.get("Status") in ("Queue", "Staging", "Running", "Killing", "Initialized")
        ]
        return active_list

    @staticmethod
    def _fetch_volc_pages(statuses: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        fields = (
            "JobId,JobName,Creator,CreateTime,LaunchTime,FinishTime,"
            "Status,Preemptible,Priority,ResourceQueueId,TaskRoleSpecs,ExitCode"
        )

        def fetch_one_page(page_idx: int) -> List[Dict[str, Any]]:
            offset = page_idx * 100
            cmd = [
                "volc", "ml_task", "list",
                "--showAllTasks",
                "--status", statuses,
                "--limit", "100",
                "--offset", str(offset),
                "--format", fields,
                "--output", "json",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=12).decode("utf-8", errors="replace")
            except Exception:
                return []
            s, e = out.find("["), out.rfind("]")
            if s == -1 or e == -1:
                return []
            try:
                page = json.loads(out[s : e + 1])
                return page if isinstance(page, list) else []
            except json.JSONDecodeError:
                return []

        results: List[Dict[str, Any]] = []
        # Fetch in parallel batches of up to 5 pages
        batch_size = min(5, max_pages)
        for start_idx in range(0, max_pages, batch_size):
            indices = list(range(start_idx, min(max_pages, start_idx + batch_size)))
            with ThreadPoolExecutor(max_workers=len(indices)) as pool:
                pages = list(pool.map(fetch_one_page, indices))
            stop = False
            for p in pages:
                results.extend(p)
                if len(p) < 100:
                    stop = True
            if stop:
                break
        return results

    @staticmethod
    def _fetch_single_task(job_id: str) -> Optional[Dict[str, Any]]:
        fields = (
            "JobId,JobName,Creator,CreateTime,LaunchTime,FinishTime,"
            "Status,Preemptible,Priority,ResourceQueueId,TaskRoleSpecs,ExitCode"
        )
        cmd = [
            "volc", "ml_task", "get",
            "--id", job_id,
            "--format", fields,
            "--output", "json",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode("utf-8", errors="replace")
            s, e = out.find("["), out.rfind("]")
            if s != -1 and e != -1:
                arr = json.loads(out[s : e + 1])
                if arr:
                    return arr[0]
        except Exception:
            pass
        return None

    def log_plan_decisions(
        self,
        plan_id: str,
        timestamp: str,
        beta: float,
        decisions: Sequence[AllocationDecision],
    ) -> int:
        now_utc = dt.datetime.now(dt.timezone.utc)
        inserted = 0
        with self._connect() as conn:
            for d in decisions:
                # Do not log HOLD on DAG deps if nothing to allocate yet
                if d.to_queue == "none" and d.from_queue == "none":
                    continue

                clean_name = d.task_id.split(" [")[0].strip()
                # Deduplication / Debounce: if same recommendation was logged within 5 minutes for this job and not yet reconciled
                recent = conn.execute(
                    """
                    SELECT audit_id, timestamp, recommended_queue
                    FROM plan_audit_history
                    WHERE job_name = ? AND reconciled = 0
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (clean_name,),
                ).fetchone()

                if recent:
                    recent_dt = self.parse_utc(recent["timestamp"])
                    if recent_dt and (now_utc - recent_dt).total_seconds() < 300.0:
                        if recent["recommended_queue"] == d.to_queue:
                            continue

                est_cost = 0.0
                if d.to_queue == "dev":
                    est_cost = d.cost_dev_min or 0.0
                elif d.to_queue == "proj":
                    est_cost = d.cost_proj_min or 0.0
                elif d.to_queue == "pre":
                    est_cost = d.cost_pre_min or 0.0

                conn.execute(
                    """
                    INSERT INTO plan_audit_history (
                        plan_id, timestamp, task_id, job_name, gpus,
                        from_queue, recommended_queue, est_cost_min,
                        proj_cost_min, pre_cost_min, beta, action_label, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan_id,
                        timestamp,
                        d.task_id,
                        clean_name,
                        d.gpus,
                        d.from_queue,
                        d.to_queue,
                        est_cost,
                        d.cost_proj_min,
                        d.cost_pre_min,
                        beta,
                        d.action_label,
                        d.reason,
                    ),
                )
                inserted += 1
        return inserted

    def reconcile_audits(self, user_id: str = DEFAULT_USER_ID) -> int:
        """
        Auto-match pending plan audits against actual volc task history:
        1. Match by job_name / task_id created around or after audit timestamp.
        2. Detect whether the agent adopted the recommended queue:
           - Remote queues: 'pre' (preemptible=1) vs 'proj' (preemptible=0)
           - Local dev: recommended 'dev' and no remote task was submitted
           - Migrations (e.g. proj -> pre): check if old job was Killed and new preemptible job launched
        3. Record ground truth wait_min, run_hrs, was_preempted, and status.
        """
        now_utc = dt.datetime.now(dt.timezone.utc)
        now_iso = now_utc.strftime("%Y-%m-%d %H:%M:%S")
        updated = 0

        with self._connect() as conn:
            pending = conn.execute(
                """
                SELECT * FROM plan_audit_history
                WHERE reconciled < 2
                ORDER BY timestamp ASC
                """
            ).fetchall()

            if not pending:
                return 0

            my_tasks = conn.execute(
                """
                SELECT * FROM task_history
                WHERE creator = ?
                ORDER BY create_time DESC
                """,
                (str(user_id),),
            ).fetchall()

            for audit in pending:
                a_id = audit["audit_id"]
                jname = audit["job_name"]
                rec_q = audit["recommended_queue"]
                from_q = audit["from_queue"]
                a_time = self.parse_utc(audit["timestamp"])

                matched_candidates = [
                    r for r in my_tasks
                    if r["job_name"] == jname
                    or r["job_name"].startswith(jname)
                    or jname.startswith(r["job_name"])
                ]

                orig_job_id = None
                m_jid = re.search(r"\[(t-[a-zA-Z0-9_-]+)\]", audit["task_id"])
                if m_jid:
                    orig_job_id = m_jid.group(1)

                orig_job = next((r for r in my_tasks if r["job_id"] == orig_job_id), None) if orig_job_id else None

                # Scenario 1: Migration recommendation (e.g. proj -> pre)
                if from_q in ("proj", "pre") and rec_q != from_q:
                    newer_job = None
                    for c in matched_candidates:
                        c_ctime = self.parse_utc(c["create_time"])
                        if c_ctime and a_time and c_ctime >= a_time - dt.timedelta(minutes=5):
                            if orig_job and c["job_id"] == orig_job["job_id"]:
                                continue
                            newer_job = c
                            break

                    if newer_job:
                        act_q = "pre" if newer_job["preemptible"] else "proj"
                        adopted = 1 if act_q == rec_q else 0
                        rec_st = 2 if newer_job["status"] in ("Success", "Failed", "Killed") else 1
                        conn.execute(
                            """
                            UPDATE plan_audit_history SET
                                actual_job_id = ?,
                                actual_queue = ?,
                                adopted = ?,
                                actual_wait_min = ?,
                                actual_run_hrs = ?,
                                was_preempted = ?,
                                status = ?,
                                reconciled = ?,
                                reconciled_at = ?
                            WHERE audit_id = ?
                            """,
                            (
                                newer_job["job_id"],
                                act_q,
                                adopted,
                                round(float(newer_job["wait_sec"]) / 60.0, 1),
                                round(float(newer_job["run_sec"]) / 3600.0, 2),
                                newer_job["preempted"],
                                newer_job["status"],
                                rec_st,
                                now_iso,
                                a_id,
                            ),
                        )
                        updated += 1
                        continue
                    elif orig_job:
                        act_q = from_q
                        adopted = 0
                        rec_st = 2 if orig_job["status"] in ("Success", "Failed", "Killed") else 1
                        conn.execute(
                            """
                            UPDATE plan_audit_history SET
                                actual_job_id = ?,
                                actual_queue = ?,
                                adopted = ?,
                                actual_wait_min = ?,
                                actual_run_hrs = ?,
                                was_preempted = ?,
                                status = ?,
                                reconciled = ?,
                                reconciled_at = ?
                            WHERE audit_id = ?
                            """,
                            (
                                orig_job["job_id"],
                                act_q,
                                adopted,
                                round(float(orig_job["wait_sec"]) / 60.0, 1),
                                round(float(orig_job["run_sec"]) / 3600.0, 2),
                                orig_job["preempted"],
                                orig_job["status"],
                                rec_st,
                                now_iso,
                                a_id,
                            ),
                        )
                        updated += 1
                        continue

                # Scenario 2: Initial placement from none
                target_job = None
                for c in matched_candidates:
                    c_ctime = self.parse_utc(c["create_time"])
                    if c_ctime and a_time and c_ctime >= a_time - dt.timedelta(minutes=15):
                        target_job = c
                        break
                if not target_job and matched_candidates:
                    target_job = matched_candidates[0]

                if target_job:
                    act_q = "pre" if target_job["preemptible"] else "proj"
                    adopted = 1 if act_q == rec_q else 0
                    rec_st = 2 if target_job["status"] in ("Success", "Failed", "Killed") else 1
                    conn.execute(
                        """
                        UPDATE plan_audit_history SET
                            actual_job_id = ?,
                            actual_queue = ?,
                            adopted = ?,
                            actual_wait_min = ?,
                            actual_run_hrs = ?,
                            was_preempted = ?,
                            status = ?,
                            reconciled = ?,
                            reconciled_at = ?
                        WHERE audit_id = ?
                        """,
                        (
                            target_job["job_id"],
                            act_q,
                            adopted,
                            round(float(target_job["wait_sec"]) / 60.0, 1),
                            round(float(target_job["run_sec"]) / 3600.0, 2),
                            target_job["preempted"],
                            target_job["status"],
                            rec_st,
                            now_iso,
                            a_id,
                        ),
                    )
                    updated += 1
                elif rec_q == "dev" and a_time:
                    age_min = (now_utc - a_time).total_seconds() / 60.0
                    if age_min > 5.0 and audit["actual_queue"] is None:
                        conn.execute(
                            """
                            UPDATE plan_audit_history SET
                                actual_queue = 'dev',
                                adopted = 1,
                                status = 'LocalDev',
                                reconciled = 1,
                                reconciled_at = ?
                            WHERE audit_id = ?
                            """,
                            (now_iso, a_id),
                        )
                        updated += 1
        return updated

    def get_audit_records(self, limit: int = 20) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM plan_audit_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def get_audit_summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM plan_audit_history").fetchall()
        if not rows:
            return {"total_decisions": 0}

        total = len(rows)
        reconciled = [r for r in rows if r["reconciled"] > 0]
        adopted_rows = [r for r in reconciled if r["adopted"] == 1]
        unadopted_rows = [r for r in reconciled if r["adopted"] == 0]

        adoption_rate = (len(adopted_rows) / len(reconciled) * 100.0) if reconciled else 0.0

        wait_errors: List[float] = []
        for r in adopted_rows:
            if r["actual_wait_min"] is not None and r["est_cost_min"] is not None and r["actual_queue"] in ("proj", "pre"):
                wait_errors.append(abs(float(r["actual_wait_min"]) - float(r["est_cost_min"])))
        mae_wait = (sum(wait_errors) / len(wait_errors)) if wait_errors else 0.0

        pre_runs = [r for r in reconciled if r["actual_queue"] == "pre"]
        preempted_runs = [r for r in pre_runs if r["was_preempted"] == 1]

        return {
            "total_decisions": total,
            "reconciled_decisions": len(reconciled),
            "adopted_decisions": len(adopted_rows),
            "unadopted_decisions": len(unadopted_rows),
            "adoption_rate_pct": round(adoption_rate, 1),
            "wait_mae_min": round(mae_wait, 1),
            "pre_total": len(pre_runs),
            "pre_preempted": len(preempted_runs),
        }

    def estimate_priors(self, queue_id: str = DEFAULT_QUEUE_ID, gpu_buckets: Sequence[int] = (1, 2, 4, 8)) -> Dict[int, QueuePriors]:
        """
        Estimate queue wait & preemption parameters for each GPU bucket g in {1, 2, 4, 8}
        by combining SQLite historical observations with Bayesian shrinkage priors and
        enforcing monotonicity across GPU counts (larger g -> >= wait and >= preemption risk).
        """
        now_utc = dt.datetime.now(dt.timezone.utc)
        priors_map: Dict[int, QueuePriors] = {}
        with self._connect() as conn:
            all_rows = conn.execute(
                "SELECT * FROM task_history WHERE queue_id = ?", (queue_id,)
            ).fetchall()

        # Total currently visible Running GPUs in `proj` (non-preemptible)
        proj_visible_running_gpus = sum(
            int(r["gpus"])
            for r in all_rows
            if r["preemptible"] == 0 and r["status"] in ("Running", "Staging")
        )

        for g in sorted(gpu_buckets):
            base = self.DEFAULT_PRIORS.get(g, self.DEFAULT_PRIORS[8])
            proj_rows = [r for r in all_rows if r["preemptible"] == 0 and r["gpus"] == g]
            pre_rows = [r for r in all_rows if r["preemptible"] == 1 and r["gpus"] == g]

            # Real-time active Queue tasks in `proj` of size <= g (if a smaller task is stuck, g is definitely stuck!)
            active_proj_q = [
                r for r in all_rows
                if r["preemptible"] == 0 and r["status"] == "Queue" and r["gpus"] <= g
            ]
            active_q_count = len(active_proj_q)
            max_q_wait = max((float(r["wait_sec"]) for r in active_proj_q), default=0.0)

            # Recent launch wait times in the last 3 hours for proj tasks of size <= g
            recent_waits: List[float] = []
            for r in all_rows:
                if r["preemptible"] == 0 and r["gpus"] <= g and r["launch_time"]:
                    lt = self.parse_utc(r["launch_time"])
                    if lt is not None and (now_utc - lt).total_seconds() <= 3 * 3600.0:
                        recent_waits.append(float(r["wait_sec"]))
            recent_launch_wait_sec = (sum(recent_waits) / len(recent_waits)) if recent_waits else 0.0

            # Split proj wait times into Light (< 20 min and uncensored) vs Heavy (>= 20 min or stuck)
            threshold_sec = 1200.0
            light_waits = [float(r["wait_sec"]) for r in proj_rows if r["wait_censored"] == 0 and float(r["wait_sec"]) < threshold_sec]
            heavy_waits = [float(r["wait_sec"]) for r in proj_rows if float(r["wait_sec"]) >= threshold_sec]

            prior_weight = 4.0
            mu_L = (prior_weight * base["proj_mu_L"] + sum(light_waits)) / (prior_weight + len(light_waits))
            n_uncens_H = sum(1 for r in proj_rows if float(r["wait_sec"]) >= threshold_sec and r["wait_censored"] == 0)
            mu_H = (prior_weight * base["proj_mu_H"] + sum(heavy_waits)) / (prior_weight + max(1, n_uncens_H))
            mu_H = max(mu_L * 3.0, mu_H)

            p0_light = (prior_weight * base["proj_p0"] + len(light_waits)) / (
                prior_weight + len(light_waits) + len(heavy_waits)
            )

            # Preemptible initial wait & requeue wait
            pre_init_waits = [float(r["wait_sec"]) for r in pre_rows if r["is_requeue"] == 0]
            pre_req_waits = [float(r["wait_sec"]) for r in pre_rows if r["is_requeue"] == 1]

            pre_mu_wait = (prior_weight * base["pre_wait"] + sum(pre_init_waits)) / (prior_weight + len(pre_init_waits))
            if len(pre_init_waits) >= 2:
                var_w = sum((w - pre_mu_wait) ** 2 for w in pre_init_waits) / len(pre_init_waits)
                pre_std_wait = max(60.0, math.sqrt(var_w))
            else:
                pre_std_wait = 0.6 * pre_mu_wait

            pre_requeue = (prior_weight * base["pre_requeue"] + sum(pre_req_waits)) / (prior_weight + len(pre_req_waits))

            # Right-censored Poisson MLE for preemption rate lambda_pre(g)
            total_pre_run_hrs = sum(float(r["run_sec"]) for r in pre_rows) / 3600.0
            total_preemptions = sum(int(r["preempted"]) for r in pre_rows)
            prior_hrs = 12.0
            pre_lambda = (prior_hrs * base["pre_lambda"] + total_preemptions) / (prior_hrs + total_pre_run_hrs)

            # Use a consistent reference runtime horizon per bucket so sparse buckets don't invert costs
            avg_run_hrs = base["avg_run_hrs"]

            priors_map[g] = QueuePriors(
                gpus=g,
                proj_mu_light_sec=mu_L,
                proj_mu_heavy_sec=mu_H,
                proj_p0_light=p0_light,
                proj_active_queued_count=active_q_count,
                proj_max_queued_wait_sec=max_q_wait,
                proj_recent_launch_wait_sec=recent_launch_wait_sec,
                proj_visible_running_gpus=proj_visible_running_gpus,
                pre_mu_wait_sec=pre_mu_wait,
                pre_std_wait_sec=pre_std_wait,
                pre_lambda_per_hr=pre_lambda,
                pre_requeue_wait_sec=pre_requeue,
                avg_run_hrs=avg_run_hrs,
                sample_count_proj=len(proj_rows),
                sample_count_pre=len(pre_rows),
            )

        # Enforce physical monotonicity across GPU sizes (g_prev < g_curr):
        # Larger GPU counts are always at least as hard to allocate / prone to preemption as smaller ones.
        sorted_gs = sorted(priors_map.keys())
        for idx in range(1, len(sorted_gs)):
            prev_p = priors_map[sorted_gs[idx - 1]]
            curr_p = priors_map[sorted_gs[idx]]
            curr_p.proj_mu_light_sec = max(curr_p.proj_mu_light_sec, prev_p.proj_mu_light_sec)
            curr_p.proj_mu_heavy_sec = max(curr_p.proj_mu_heavy_sec, prev_p.proj_mu_heavy_sec)
            curr_p.proj_p0_light = min(curr_p.proj_p0_light, prev_p.proj_p0_light)
            curr_p.pre_mu_wait_sec = max(curr_p.pre_mu_wait_sec, prev_p.pre_mu_wait_sec)
            curr_p.pre_lambda_per_hr = max(curr_p.pre_lambda_per_hr, prev_p.pre_lambda_per_hr)

        return priors_map


# ============================================================================
# 3. Local DevInstance GPU Detector (`nvidia-smi`)
# ============================================================================

def detect_dev_gpus(reserved_gpus: int = 0) -> Tuple[int, int]:
    """
    Detect (total_gpus, free_gpus) on the current devinstance via `nvidia-smi`.
    A GPU is considered free if its memory usage is < 1024 MiB and utilization < 15%.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        total = len(lines)
        idle = 0
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                mem_mib = float(parts[1])
                util_pct = float(parts[2])
                if mem_mib < 1024.0 and util_pct < 15.0:
                    idle += 1
        free = max(0, idle - max(0, reserved_gpus))
        return total, free
    except Exception:
        return 0, 0


# ============================================================================
# 4. Core Black-Box Solver: M = f(g, s, DAG; beta)
# ============================================================================

class DynamicAllocator:
    """
    Solves for the optimal allocation matrix M in {0, 1}^{N x 4} over queues
    {'none', 'dev', 'proj', 'pre'} given task states {(g_i, s_i)} and DAG.
    """

    def __init__(self, proj_self_congestion_sec_per_gpu: float = 135.0, switch_hysteresis_sec: float = 120.0) -> None:
        self.kappa_cong = proj_self_congestion_sec_per_gpu
        self.hysteresis_sec = switch_hysteresis_sec

    @staticmethod
    def compute_dag_descendants(tasks: Sequence[TaskState]) -> Dict[str, int]:
        """Count how many downstream tasks in the DAG depend (transitively) on each task."""
        children: Dict[str, Set[str]] = {t.task_id: set() for t in tasks}
        for t in tasks:
            for dep in t.deps:
                if dep in children:
                    children[dep].add(t.task_id)

        desc_count: Dict[str, int] = {}
        for t in tasks:
            visited: Set[str] = set()
            stack = list(children.get(t.task_id, ()))
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(children.get(curr, ()))
            desc_count[t.task_id] = len(visited)
        return desc_count

    def solve(
        self,
        tasks: Sequence[TaskState],
        dev_total_gpus: int,
        dev_free_gpus: int,
        priors_by_gpu: Dict[int, QueuePriors],
        beta: float = 0.3,
        active_running_tasks: Optional[Sequence[str]] = None,
        completed_tasks: Optional[Sequence[str]] = None,
    ) -> AllocationPlanResult:
        """
        Solve the integer quadratic allocation problem:
            min_M  sum_{i in Ready} (L_{i,dev} M_{i,dev} + L_{i,proj} M_{i,proj} + L_{i,pre} M_{i,pre})
                   + (kappa_cong / 2) * (sum_{i in Ready} g_i M_{i,proj})^2
            s.t.   M_{i,none} = 1                         for all i in Blocked
                   M_{i,dev} + M_{i,proj} + M_{i,pre} = 1 for all i in Ready
                   sum_{i in Ready} g_i M_{i,dev} <= dev_free_gpus
        Solved to exact global optimality via Dynamic Programming over (dev_gpus_used, proj_gpus_used).
        """
        completed_set = set(completed_tasks or ())
        desc_map = self.compute_dag_descendants(tasks)

        ready_tasks: List[TaskState] = []
        blocked_tasks: List[TaskState] = []
        for t in tasks:
            t.dag_descendants = desc_map.get(t.task_id, 0)
            unresolved_deps = [d for d in t.deps if d not in completed_set]
            if unresolved_deps:
                blocked_tasks.append(t)
            else:
                ready_tasks.append(t)

        # Precompute per-task costs L_{i,j} (in seconds) for ready tasks
        costs: Dict[str, Dict[QueueName, float]] = {}
        for t in ready_tasks:
            p = priors_by_gpu.get(t.gpus) or priors_by_gpu[max(priors_by_gpu.keys())]
            # 1. Cost on `dev`: 0 wait, 0 preemption, plus negative reward to prioritize high-cost/critical tasks on dev
            c_proj_raw = p.proj_residual_wait_cost(t.waited_sec if t.current_queue == "proj" else 0.0, beta)
            c_pre_raw = (
                p.pre_residual_wait_cost(t.waited_sec if t.current_queue == "pre" else 0.0, beta)
                + p.pre_preemption_penalty_sec(beta, dag_descendants=t.dag_descendants)
            )
            # Slight hysteresis to avoid unnecessary queue hopping on tiny cost differences
            c_proj = c_proj_raw + (0.0 if t.current_queue in ("proj", "none") else self.hysteresis_sec)
            c_pre = c_pre_raw + (0.0 if t.current_queue in ("pre", "none") else self.hysteresis_sec)

            # If task fits on dev, its effective latency is 0 (with slight tie-breaking bonus for critical/heavy tasks)
            c_dev = -1.0 * (0.01 * t.gpus + 0.05 * t.dag_descendants) if t.gpus <= dev_free_gpus else float("inf")
            costs[t.task_id] = {"dev": c_dev, "proj": c_proj, "pre": c_pre, "none": float("inf")}

        # Exact Dynamic Programming over ready_tasks:
        # State: (dev_gpus_used, proj_gpus_used) -> (min_linear_cost, backpointer_list)
        dp: Dict[Tuple[int, int], Tuple[float, List[QueueName]]] = {(0, 0): (0.0, [])}

        for t in ready_tasks:
            c_map = costs[t.task_id]
            next_dp: Dict[Tuple[int, int], Tuple[float, List[QueueName]]] = {}
            for (d_used, p_used), (acc_cost, choices) in dp.items():
                # Option 1: assign to `dev` (if capacity permits)
                if d_used + t.gpus <= dev_free_gpus:
                    ns = (d_used + t.gpus, p_used)
                    nc = acc_cost + c_map["dev"]
                    if ns not in next_dp or nc < next_dp[ns][0]:
                        next_dp[ns] = (nc, choices + ["dev"])
                # Option 2: assign to `proj` (subtract diagonal 0.5*kappa*g_i^2 so self-congestion only penalizes cross-terms sum_{i<k} g_i g_k)
                ns_proj = (d_used, p_used + t.gpus)
                nc_proj = acc_cost + c_map["proj"] - 0.5 * self.kappa_cong * (t.gpus ** 2)
                if ns_proj not in next_dp or nc_proj < next_dp[ns_proj][0]:
                    next_dp[ns_proj] = (nc_proj, choices + ["proj"])
                # Option 3: assign to `pre`
                ns_pre = (d_used, p_used)
                nc_pre = acc_cost + c_map["pre"]
                if ns_pre not in next_dp or nc_pre < next_dp[ns_pre][0]:
                    next_dp[ns_pre] = (nc_pre, choices + ["pre"])
            dp = next_dp

        # Pick state minimizing total linear cost + quadratic proj self-congestion penalty
        best_total = float("inf")
        best_choices: List[QueueName] = []
        for (d_used, p_used), (lin_cost, choices) in dp.items():
            cong_penalty = 0.5 * self.kappa_cong * (p_used ** 2)
            total_obj = lin_cost + cong_penalty
            if total_obj < best_total:
                best_total = total_obj
                best_choices = choices

        ready_assignment: Dict[str, QueueName] = {
            t.task_id: q_choice for t, q_choice in zip(ready_tasks, best_choices)
        }

        # Assemble final matrix M and human-readable decisions in original task order
        matrix_M: Dict[str, Dict[QueueName, int]] = {}
        decisions: List[AllocationDecision] = []

        for t in tasks:
            if t.task_id in ready_assignment:
                target_q = ready_assignment[t.task_id]
                c_map = costs[t.task_id]
                c_dev_m = 0.0 if t.gpus <= dev_free_gpus else None
                c_proj_m = round(c_map["proj"] / 60.0, 1)
                c_pre_m = round(c_map["pre"] / 60.0, 1)
            else:
                target_q = "none"
                c_dev_m = None
                c_proj_m = None
                c_pre_m = None

            row_m: Dict[QueueName, int] = {q: (1 if q == target_q else 0) for q in QUEUE_COLS}
            matrix_M[t.task_id] = row_m

            action_label, reason = self._describe_transition(
                task=t,
                to_q=target_q,
                c_proj_min=c_proj_m,
                c_pre_min=c_pre_m,
                completed_set=completed_set,
            )
            decisions.append(
                AllocationDecision(
                    task_id=t.task_id,
                    gpus=t.gpus,
                    from_queue=t.current_queue,
                    to_queue=target_q,
                    waited_min=round(t.waited_sec / 60.0, 1),
                    row_M=row_m,
                    cost_dev_min=c_dev_m,
                    cost_proj_min=c_proj_m,
                    cost_pre_min=c_pre_m,
                    action_label=action_label,
                    reason=reason,
                )
            )

        return AllocationPlanResult(
            timestamp=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            beta=beta,
            dev_total_gpus=dev_total_gpus,
            dev_free_gpus=dev_free_gpus,
            active_running_tasks=list(active_running_tasks or ()),
            completed_tasks=list(completed_tasks or ()),
            matrix_M=matrix_M,
            decisions=decisions,
            priors_by_gpu=priors_by_gpu,
        )

    @staticmethod
    def _describe_transition(
        task: TaskState,
        to_q: QueueName,
        c_proj_min: Optional[float],
        c_pre_min: Optional[float],
        completed_set: Set[str],
    ) -> Tuple[str, str]:
        from_q = task.current_queue
        if to_q == "none":
            unresolved = [d for d in task.deps if d not in completed_set]
            return "HOLD (none)", f"Waiting on DAG deps: {', '.join(unresolved)}"

        if from_q == to_q:
            if to_q == "proj":
                return f"KEEP ({to_q})", f"Stay queued in proj (rem cost ~{c_proj_min}m < pre ~{c_pre_min}m)"
            if to_q == "pre":
                return f"KEEP ({to_q})", f"Stay queued in pre (cost ~{c_pre_min}m <= proj ~{c_proj_min}m)"
            return f"KEEP ({to_q})", f"Keep in {to_q}"

        if from_q == "none":
            if to_q == "dev":
                return "none -> dev", f"DevInstance has {task.gpus} free GPUs (0 wait, 0 preemption)"
            if to_q == "proj":
                return "none -> proj", f"Submit non-preemptible (proj cost ~{c_proj_min}m vs pre ~{c_pre_min}m)"
            return "none -> pre", f"Submit preemptible (pre cost ~{c_pre_min}m vs proj ~{c_proj_min}m + congestion)"

        if to_q == "dev":
            return f"{from_q} -> dev", f"Harvest queued task to free local DevInstance ({task.gpus} GPUs)"
        if from_q == "proj" and to_q == "pre":
            return "proj -> pre", (
                f"Proj queue timeout after {task.waited_sec/60.0:.1f}m "
                f"(proj rem cost ~{c_proj_min}m > pre cost ~{c_pre_min}m)"
            )
        if from_q == "pre" and to_q == "proj":
            return "pre -> proj", f"Proj pool opened up (proj cost ~{c_proj_min}m < pre cost ~{c_pre_min}m)"
        return f"{from_q} -> {to_q}", f"Move from {from_q} to {to_q}"


# ============================================================================
# 5. Task Input Loader & Online State Matcher (`my_tasks.sh` integration)
# ============================================================================

def load_tasks_and_match_online(
    tasks_file: Optional[Path],
    pending_gpus_arg: Optional[str],
    db: TelemetryDB,
    queue_id: str,
    user_id: str,
    do_sync: bool = True,
) -> Tuple[List[TaskState], List[str], List[str]]:
    """
    Load unstarted tasks from YAML/CLI and match against `volc ml_task list` (and SQLite history)
    for `Creator == user_id` to automatically populate (q_i, t_i), running tasks, and completed DAG deps.
    """
    now_utc = dt.datetime.now(dt.timezone.utc)
    if do_sync and db.needs_sync(ttl_sec=90.0):
        db.sync_from_volc(queue_id=queue_id, user_id=user_id)

    with db._connect() as conn:
        my_rows = conn.execute(
            "SELECT * FROM task_history WHERE creator = ? ORDER BY create_time DESC",
            (str(user_id),),
        ).fetchall()

    # Keep only the latest record per job_name for our user
    latest_by_name: Dict[str, sqlite3.Row] = {}
    by_job_id: Dict[str, sqlite3.Row] = {}
    for r in my_rows:
        by_job_id[r["job_id"]] = r
        if r["job_name"] not in latest_by_name:
            latest_by_name[r["job_name"]] = r

    active_running: List[str] = []
    completed_ids: List[str] = []
    unstarted_tasks: List[TaskState] = []

    if tasks_file is not None:
        raw_text = Path(tasks_file).read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) if yaml is not None else json.loads(raw_text)
        raw_tasks = data.get("tasks") or data.get("workloads") or []

        for item in raw_tasks:
            tid = str(item.get("id") or item.get("name"))
            jname = str(item.get("name") or tid)
            gpus = int(item.get("gpus") or 1)
            deps = [str(d) for d in (item.get("deps") or [])]

            # Explicit manual state override in YAML if provided
            explicit_q = item.get("queue")
            explicit_wait_min = item.get("waited_min")
            explicit_status = str(item.get("status") or "").lower()

            if explicit_status in ("completed", "success", "done"):
                completed_ids.append(tid)
                continue
            if explicit_status in ("running", "staging"):
                active_running.append(tid)
                continue

            matched = by_job_id.get(str(item.get("job_id") or "")) or latest_by_name.get(jname) or latest_by_name.get(tid)

            if explicit_q is not None:
                q_name: QueueName = explicit_q if explicit_q in QUEUE_COLS else "none"
                waited_sec = float(explicit_wait_min or 0.0) * 60.0
                unstarted_tasks.append(
                    TaskState(task_id=tid, gpus=gpus, current_queue=q_name, waited_sec=waited_sec, deps=deps)
                )
            elif matched is not None:
                st = matched["status"]
                if st == "Success":
                    completed_ids.append(tid)
                elif st in ("Running", "Staging"):
                    active_running.append(f"{tid} ({matched['job_id']})")
                elif st in ("Queue", "Initialized"):
                    q_name = "pre" if matched["preemptible"] else "proj"
                    ctime = TelemetryDB.parse_utc(matched["create_time"]) or now_utc
                    waited_sec = max(0.0, (now_utc - ctime).total_seconds())
                    unstarted_tasks.append(
                        TaskState(
                            task_id=tid,
                            gpus=int(matched["gpus"] or gpus),
                            current_queue=q_name,
                            waited_sec=waited_sec,
                            deps=deps,
                            job_id=matched["job_id"],
                        )
                    )
                else:
                    # Failed / Killed -> treat as unstarted (none, t=0) so it can be re-allocated
                    unstarted_tasks.append(
                        TaskState(task_id=tid, gpus=gpus, current_queue="none", waited_sec=0.0, deps=deps)
                    )
            else:
                # Not submitted yet -> default to (queue="none", t=0)
                unstarted_tasks.append(
                    TaskState(task_id=tid, gpus=gpus, current_queue="none", waited_sec=0.0, deps=deps)
                )
    else:
        # Zero-file mode: auto-discover all currently Queued tasks of user_id from volc
        for r in my_rows:
            if r["status"] in ("Running", "Staging"):
                active_running.append(f"{r['job_name']} ({r['job_id']})")
            elif r["status"] in ("Queue", "Initialized"):
                q_name = "pre" if r["preemptible"] else "proj"
                ctime = TelemetryDB.parse_utc(r["create_time"]) or now_utc
                waited_sec = max(0.0, (now_utc - ctime).total_seconds())
                unstarted_tasks.append(
                    TaskState(
                        task_id=f"{r['job_name']} [{r['job_id']}]",
                        gpus=int(r["gpus"]),
                        current_queue=q_name,
                        waited_sec=waited_sec,
                        deps=[],
                        job_id=r["job_id"],
                    )
                )

    # Append any ad-hoc unsubmitted tasks from --pending-gpus (e.g. "2,4,8")
    if pending_gpus_arg:
        for idx, tok in enumerate(pending_gpus_arg.split(","), start=1):
            tok = tok.strip()
            if tok:
                g = int(tok)
                unstarted_tasks.append(
                    TaskState(
                        task_id=f"pending_{idx}_{g}gpu",
                        gpus=g,
                        current_queue="none",
                        waited_sec=0.0,
                        deps=[],
                    )
                )

    return unstarted_tasks, active_running, completed_ids


# ============================================================================
# 6. CLI Formatting & Entrypoint
# ============================================================================

def print_stats_table(priors: Dict[int, QueuePriors], dev_total: int, dev_free: int) -> None:
    print(f"Local DevInstance GPUs : {dev_free} free / {dev_total} total")
    print("-" * 104)
    print(
        f"{'GPUs':<6} {'Proj Light(m)':<15} {'Proj Heavy(m)':<15} {'Proj P(Light)':<15} "
        f"{'Proj Queued':<13} {'Pre Wait(m)':<13} {'Pre MTBF(h)':<13} {'Samples(P/E)'}"
    )
    print("-" * 104)
    for g, p in sorted(priors.items()):
        mtbf = 1.0 / max(1e-4, p.pre_lambda_per_hr)
        print(
            f"{g:<6} {p.proj_mu_light_sec/60.0:<15.1f} {p.proj_mu_heavy_sec/60.0:<15.1f} "
            f"{p.proj_p0_light:<15.2f} {p.proj_active_queued_count:<13} "
            f"{p.pre_mu_wait_sec/60.0:<13.1f} {mtbf:<13.2f} "
            f"{p.sample_count_proj}/{p.sample_count_pre}"
        )


def print_plan_table(res: AllocationPlanResult) -> None:
    print("=" * 116)
    print(
        f"[{res.timestamp}] Volc MLP Dynamic Allocation Plan  |  "
        f"beta={res.beta:.2f}  |  DevInstance Free GPUs: {res.dev_free_gpus}/{res.dev_total_gpus}"
    )
    if res.active_running_tasks:
        print(f"Already Running (excluded from I): {', '.join(res.active_running_tasks)}")
    if res.completed_tasks:
        print(f"Completed (unlocked DAG deps)    : {', '.join(res.completed_tasks)}")
    print("=" * 116)

    if not res.decisions:
        print("No unstarted tasks to allocate. (Pass `--tasks <yaml>` or `--pending-gpus 2,4,8`)")
        return

    print(
        f"{'Task ID':<24} {'g_i':<4} {'State s_i=(q_i, t_i)':<20} "
        f"{'M=[none,dev,proj,pre]':<23} {'Action':<16} {'Reason'}"
    )
    print("-" * 116)
    for d in res.decisions:
        state_str = f"({d.from_queue}, {d.waited_min:.1f}m)"
        m_vec = f"[{d.row_M['none']},   {d.row_M['dev']},   {d.row_M['proj']},    {d.row_M['pre']}  ]"
        tid_disp = d.task_id if len(d.task_id) <= 23 else d.task_id[:20] + "..."
        print(
            f"{tid_disp:<24} {d.gpus:<4} {state_str:<20} "
            f"{m_vec:<23} {d.action_label:<16} {d.reason}"
        )
    print("-" * 116)


def print_audit_table(records: List[sqlite3.Row], summary: Dict[str, Any]) -> None:
    print("=" * 118)
    print("Volc MLP Decision Audit Log & Telemetry")
    print("=" * 118)
    if not records:
        print("No audit records found yet. Run `./scripts/mlp_sched.py plan` to generate decisions.")
        return

    print(
        f"{'Audit ID':<10} {'Timestamp':<19} {'Task / Job':<22} {'GPUs':<5} "
        f"{'Rec':<6} {'Actual':<7} {'Adopted':<9} {'Est(m)':<8} {'Act(m)':<8} {'Preempt':<8} {'Status'}"
    )
    print("-" * 118)
    for r in records:
        adopted_str = "YES" if r["adopted"] == 1 else ("NO" if r["adopted"] == 0 else "WAITING")
        est_str = f"{r['est_cost_min']:.1f}" if r["est_cost_min"] is not None else "-"
        act_wait = f"{r['actual_wait_min']:.1f}" if r["actual_wait_min"] is not None else "-"
        preempt_str = "YES" if r["was_preempted"] == 1 else "NO"
        jdisp = r["job_name"] if len(r["job_name"]) <= 21 else r["job_name"][:18] + "..."
        act_q = r["actual_queue"] or "-"
        st = r["status"] or "Pending"
        print(
            f"#{r['audit_id']:<9} {r['timestamp'][:19]:<19} {jdisp:<22} {r['gpus']:<5} "
            f"{r['recommended_queue']:<6} {act_q:<7} {adopted_str:<9} {est_str:<8} {act_wait:<8} {preempt_str:<8} {st}"
        )
    print("-" * 118)
    if summary.get("reconciled_decisions", 0) > 0:
        print(
            f"Summary Stats: Total Tracked={summary['total_decisions']} | "
            f"Reconciled={summary['reconciled_decisions']} | "
            f"Agent Adoption Rate={summary['adoption_rate_pct']}% ({summary['adopted_decisions']}/{summary['reconciled_decisions']}) | "
            f"Wait MAE={summary['wait_mae_min']}m | "
            f"Preemptions in Pre={summary['pre_preempted']}/{summary['pre_total']}"
        )
    else:
        print(f"Summary Stats: Total Tracked={summary['total_decisions']} (Awaiting live task execution & reconciliation)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Volc MLP Dynamic Task Allocation Advisor: M = f(g, s, DAG; beta)"
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite telemetry DB path")
    parser.add_argument("--queue-id", type=str, default=DEFAULT_QUEUE_ID, help="Volc ResourceQueueID")
    parser.add_argument("--user-id", type=str, default=DEFAULT_USER_ID, help="Volc Creator IAM user ID")

    sub = parser.add_subparsers(dest="cmd")

    # 1. sync
    p_sync = sub.add_parser("sync", help="Sync queue history & status from volc into SQLite DB")
    p_sync.add_argument("--active-pages", type=int, default=10, help="Max pages (100/page) of active tasks")
    p_sync.add_argument("--terminal-pages", type=int, default=3, help="Max pages of recent terminal tasks")

    # 2. stats
    p_stats = sub.add_parser("stats", help="Show estimated queue wait & preemption parameters from DB")
    p_stats.add_argument("--no-sync", action="store_true", help="Skip live volc sync before estimating")
    p_stats.add_argument("--json", action="store_true", help="Output JSON format")

    # 3. plan
    p_plan = sub.add_parser("plan", help="Solve optimal allocation matrix M = f(g, s) for unstarted tasks")
    p_plan.add_argument("--tasks", type=Path, default=None, help="YAML/JSON file of tasks (g_i, s_i, deps)")
    p_plan.add_argument("--pending-gpus", type=str, default=None, help="Comma-separated GPUs of unsubmitted tasks (e.g. 2,4,4,8)")
    p_plan.add_argument("--beta", type=float, default=None, help="Risk aversion parameter in [0, 1] (overrides YAML default 0.3)")
    p_plan.add_argument("--dev-reserved-gpus", type=int, default=None, help="GPUs to reserve on local devinstance (default 0)")
    p_plan.add_argument("--dev-free-gpus", type=int, default=None, help="Explicitly override free local DevInstance GPUs (for simulation)")
    p_plan.add_argument("--no-sync", action="store_true", help="Offline mode: do not call volc ml_task list")
    p_plan.add_argument("--no-audit", action="store_true", help="Do not write decisions into plan_audit_history")
    p_plan.add_argument("--output", choices=["table", "json"], default="table", help="Output format")
    p_plan.add_argument("--watch", action="store_true", help="Continuously refresh telemetry and re-print plan M")
    p_plan.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds when --watch is enabled")

    # 4. audit
    p_audit = sub.add_parser("audit", help="Show decision audit history and agent adoption / accuracy stats")
    p_audit.add_argument("--limit", type=int, default=20, help="Max records to display")
    p_audit.add_argument("--no-sync", action="store_true", help="Skip live sync before checking audit log")
    p_audit.add_argument("--json", action="store_true", help="Output JSON format")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd or "plan"

    db = TelemetryDB(db_path=args.db_path)

    # Validate that queue_id and user_id are configured if live volc interaction is needed
    needs_volc = (cmd in ("sync", "audit") and not getattr(args, "no_sync", False)) or (
        cmd in ("stats", "plan") and not getattr(args, "no_sync", False)
    )
    if needs_volc and (not args.queue_id or not args.user_id):
        print(
            "Error: Resource Queue ID or User ID is missing.\n"
            "Please configure via config.local.yaml (copy from config.example.yaml),\n"
            "or set environment variables VOLC_QUEUE_ID and VOLC_USER_ID,\n"
            "or pass flags --queue-id <id> and --user-id <id>.",
            file=sys.stderr,
        )
        return 1

    if cmd == "sync":
        active = db.sync_from_volc(
            queue_id=args.queue_id,
            user_id=args.user_id,
            active_pages=args.active_pages,
            terminal_pages=args.terminal_pages,
        )
        print(f"Synced telemetry to {args.db_path} ({len(active)} active tasks tracked in queue).")
        return 0

    if cmd == "stats":
        if not args.no_sync and db.needs_sync(ttl_sec=90.0):
            db.sync_from_volc(queue_id=args.queue_id, user_id=args.user_id)
        priors = db.estimate_priors(queue_id=args.queue_id)
        dev_total, dev_free = detect_dev_gpus()
        if args.json:
            print(
                json.dumps(
                    {
                        "dev_total_gpus": dev_total,
                        "dev_free_gpus": dev_free,
                        "priors": {k: asdict(v) for k, v in priors.items()},
                    },
                    indent=2,
                )
            )
        else:
            print_stats_table(priors, dev_total, dev_free)
        return 0

    if cmd == "audit":
        if not args.no_sync and db.needs_sync(ttl_sec=90.0):
            db.sync_from_volc(queue_id=args.queue_id, user_id=args.user_id)
        else:
            db.reconcile_audits(user_id=args.user_id)

        records = db.get_audit_records(limit=args.limit)
        summary = db.get_audit_summary()
        if args.json:
            print(
                json.dumps(
                    {
                        "summary": summary,
                        "records": [dict(r) for r in records],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print_audit_table(records, summary)
        return 0

    if cmd == "plan":
        # Load config defaults from YAML if provided
        yaml_cfg: Dict[str, Any] = {}
        if args.tasks is not None and args.tasks.exists():
            raw = args.tasks.read_text(encoding="utf-8")
            parsed = yaml.safe_load(raw) if yaml is not None else json.loads(raw)
            if isinstance(parsed, dict):
                yaml_cfg = parsed.get("config") or {}

        beta = args.beta if args.beta is not None else float(yaml_cfg.get("beta", 0.3))
        dev_reserved = (
            args.dev_reserved_gpus
            if args.dev_reserved_gpus is not None
            else int(yaml_cfg.get("dev_reserved_gpus", 0))
        )

        allocator = DynamicAllocator()

        while True:
            tasks, active_running, completed_ids = load_tasks_and_match_online(
                tasks_file=args.tasks,
                pending_gpus_arg=args.pending_gpus,
                db=db,
                queue_id=args.queue_id,
                user_id=args.user_id,
                do_sync=not args.no_sync,
            )
            priors = db.estimate_priors(queue_id=args.queue_id)
            dev_total, dev_free = detect_dev_gpus(reserved_gpus=dev_reserved)
            if args.dev_free_gpus is not None:
                dev_free = max(0, args.dev_free_gpus)
                dev_total = max(dev_total, dev_free)

            result = allocator.solve(
                tasks=tasks,
                dev_total_gpus=dev_total,
                dev_free_gpus=dev_free,
                priors_by_gpu=priors,
                beta=beta,
                active_running_tasks=active_running,
                completed_tasks=completed_ids,
            )

            if not args.no_audit:
                plan_id = f"plan-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                db.log_plan_decisions(
                    plan_id=plan_id,
                    timestamp=result.timestamp,
                    beta=result.beta,
                    decisions=result.decisions,
                )

            if args.output == "json":
                print(
                    json.dumps(
                        {
                            "timestamp": result.timestamp,
                            "beta": result.beta,
                            "dev_total_gpus": result.dev_total_gpus,
                            "dev_free_gpus": result.dev_free_gpus,
                            "active_running_tasks": result.active_running_tasks,
                            "completed_tasks": result.completed_tasks,
                            "matrix_M": result.matrix_M,
                            "decisions": [asdict(d) for d in result.decisions],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                if args.watch:
                    os.system("clear" if os.name == "posix" else "cls")
                print_plan_table(result)

            if not args.watch:
                break
            time.sleep(max(5, args.interval))

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
