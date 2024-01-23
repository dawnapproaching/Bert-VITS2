#!/bin/bash

# conda 选择环境
source ~/anaconda3/bin/activate
conda activate vits

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../"
echo `pwd`
# 创建日志文件
if [ ! -d "logs" ]; then
    mkdir logs
fi
# 杀进程
pid=`ps -ef | grep "damon_server.py" | grep -v grep | awk '{print $2}'`
if [ -n "$pid" ]; then
    kill -9 $pid
    echo "kill pid: $pid"
fi
cd "../"
# 启动进程
nohup python3 server/damon_server.py > server/logs/server.log 2>&1 &

tail -F server/logs/server.log
