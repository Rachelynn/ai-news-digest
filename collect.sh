#!/bin/bash
# AI新闻收集与摘要脚本
# 每8小时运行一次

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
DATE=$(date +"%Y-%m-%d")
DATETIME=$(date +"%Y-%m-%d_%H-%M")

echo "🤖 开始收集AI新闻 - $DATETIME"

# 创建日志目录
mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_DIR/summaries"

LOG_FILE="$WORK_DIR/logs/collect_$DATETIME.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 收集新闻的Python脚本
python3 "$WORK_DIR/collect_news.py" --output "$WORK_DIR/summaries/${DATE}_${DATETIME}.md"

# Git提交并推送
if [ -d "$WORK_DIR/.git" ]; then
    cd "$WORK_DIR"
    git add .
    git commit -m "AI新闻更新: $DATETIME" || true
    git push origin main || git push origin master || echo "推送失败，请检查远程仓库配置"
fi

echo "✅ 完成 - $DATETIME"
