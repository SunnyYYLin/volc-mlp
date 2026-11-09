#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <task_id> [lines] [keyword] [instance]" >&2
    echo "示例:" >&2
    echo "  $0 t-20261015143939-bntlh              # 查看默认最新 100 行日志" >&2
    echo "  $0 t-20261015143939-bntlh 200          # 查看最新 200 行" >&2
    echo "  $0 t-20261015143939-bntlh 500 Error    # 过滤包含 Error 的日志" >&2
    echo "  $0 t-20261015143939-bntlh 100 '' worker_1 # 查看指定 worker 节点" >&2
    exit 1
fi

TASK_ID="$1"
LINES="${2:-100}"
KEYWORD="${3:-}"
INSTANCE="${4:-worker_0}"

# 自动纠正实例名：如果传入了 Pod 全名 (如 t-xxx-worker-0)，自动转换为 worker_0
if [[ "$INSTANCE" =~ worker[-_]([0-9]+)$ ]]; then
    INSTANCE="worker_${BASH_REMATCH[1]}"
fi

CMD=(volc ml_task logs -t "$TASK_ID" -i "$INSTANCE" -l "$LINES")

if [[ -n "$KEYWORD" ]]; then
    CMD+=(-c "$KEYWORD")
fi

exec "${CMD[@]}"
