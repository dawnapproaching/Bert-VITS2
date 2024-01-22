#!/bin/bash

# conda 选择环境
source ~/anaconda3/bin/activate
conda activate vits
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
# 启动进程
nohup python3 damon_server.py > logs/server.log 2>&1 &
