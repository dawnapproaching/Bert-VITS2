#!/bin/bash
pid=`ps -ef | grep "damon_server.py" | grep -v grep | awk '{print $2}'`
if [ -n "$pid" ]; then
    kill -9 $pid
fi