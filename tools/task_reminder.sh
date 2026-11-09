#!/bin/bash
# task_reminder.sh - 监控任务状态并在状态改变或超时时退出
# Usage: ./task_reminder.sh <job_id> [timeout_seconds] [interval_seconds]

if [ -z "$1" ]; then
    echo "用法: $0 <job_id> [timeout_seconds] [interval_seconds]"
    exit 1
fi

JOB_ID=$1
TIMEOUT=${2:-300}
INTERVAL=${3:-10}

get_job_state() {
    # 忽略 stderr 中的 metrics 报错，并在 stdout 中找到 State 字段
    volc ml_task get --id "$JOB_ID" --output json 2>/dev/null | grep -o '"Status":"[^"]*"' | head -n 1 | cut -d'"' -f4
}

echo "======================================================="
echo "开始监控任务: $JOB_ID"
echo "超时时间: ${TIMEOUT}s, 轮询间隔: ${INTERVAL}s"
echo "======================================================="

INITIAL_STATE=$(get_job_state)
if [ -z "$INITIAL_STATE" ]; then
    echo "错误: 无法获取任务 $JOB_ID 的初始状态。请检查任务 ID 是否正确。"
    exit 1
fi

echo "初始状态: $INITIAL_STATE"
echo "开始轮询..."

ELAPSED=0

while true; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    
    CURRENT_STATE=$(get_job_state)
    
    if [ -z "$CURRENT_STATE" ]; then
        echo -e "\n警告: 无法获取当前状态，可能遇到网络网络问题，重试中..."
        continue
    fi

    if [ "$CURRENT_STATE" != "$INITIAL_STATE" ]; then
        echo -e "\n[!] 任务状态已改变: $INITIAL_STATE -> $CURRENT_STATE"
        echo "监控结束。"
        exit 0
    fi
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\n[!] 已达到超时时间 (${TIMEOUT}s)。任务状态仍为: $INITIAL_STATE"
        echo "======================================================="
        echo "❗️❗️ 提示模型 (ATTENTION MODEL) ❗️❗️"
        echo "任务状态尚未改变，且已达到单次监控超时限制。"
        echo "为了防止阻塞，当前脚本已自动退出。"
        echo "请模型记得继续调用以下命令以继续监控该任务："
        echo "./task_reminder.sh $JOB_ID $TIMEOUT $INTERVAL"
        echo "======================================================="
        exit 0
    fi
    
    # 打印进度并立即刷新
    echo -n "."
done
