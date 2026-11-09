#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

USER_ID="${VOLC_USER_ID:-}"
QUEUE_ID="${VOLC_QUEUE_ID:-}"

if [[ -z "$USER_ID" || -z "$QUEUE_ID" ]] && [[ -f "$SCRIPT_DIR/config.local.yaml" ]]; then
    eval "$(python3 -c "
import yaml
try:
    c = yaml.safe_load(open('$SCRIPT_DIR/config.local.yaml')) or {}
    uid = c.get('user_id', '')
    qid = c.get('queue_id', '')
    if not '$USER_ID':
        print(f'USER_ID=\"{uid}\"')
    if not '$QUEUE_ID':
        print(f'QUEUE_ID=\"{qid}\"')
except Exception:
    pass
")"
fi

if [[ -z "$USER_ID" || -z "$QUEUE_ID" ]]; then
    echo "错误: 请通过环境变量 (VOLC_USER_ID, VOLC_QUEUE_ID) 或 config.local.yaml 设置账号与队列。" >&2
    exit 1
fi
STATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state)
            STATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
done

STATUSES="Queue,Staging,Running,Killing,Success,Failed,Killed,Initialized"
if [[ -n "$STATE" ]]; then
    STATUSES="$STATE"
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

for offset in $(seq 0 100 900); do
    page_file="$TEMP_DIR/$offset.json"
    volc ml_task list \
        --status "$STATUSES" \
        --limit 100 \
        --offset "$offset" \
        --output json \
        2>/dev/null \
        | grep -v '^\[Metrics' \
        >"$page_file"
    page_count=$(python3 -c '
import json, sys
s = open(sys.argv[1], encoding="utf-8").read()
start, end = s.find("["), s.rfind("]")
print(len(json.loads(s[start:end+1])) if start != -1 and end != -1 else 0)
' "$page_file")
    if [[ "$page_count" -lt 100 ]]; then
        break
    fi
done

python3 - "$TEMP_DIR" "$USER_ID" "$QUEUE_ID" <<'PY'
import json
import sys
from pathlib import Path

directory, user_id, queue_id = sys.argv[1:]
tasks = []
for path in sorted(Path(directory).glob("*.json"), key=lambda item: int(item.stem)):
    with path.open(encoding="utf-8") as source:
        content = source.read()
        start, end = content.find("["), content.rfind("]")
        if start != -1 and end != -1:
            tasks.extend(json.loads(content[start:end+1]))

tasks = [
    task
    for task in tasks
    if str(task.get("Creator")) == user_id and task.get("ResourceQueueId") == queue_id
]

print(f"找到 {len(tasks)} 个任务:\n")
print(f"{'ID':<25} {'Name':<30} {'State':<12} {'Start':<25}")
print("-" * 96)
for task in tasks:
    name = str(task.get("JobName", ""))
    if len(name) > 28:
        name = name[:25] + "..."
    print(
        f"{str(task.get('JobId', '')):<25} "
        f"{name:<30} "
        f"{str(task.get('Status', '')):<12} "
        f"{str(task.get('Start') or ''):<25}"
    )
PY
