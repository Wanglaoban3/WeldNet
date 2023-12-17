@echo off
cmd "/c activate pytorch && python ssl_train.py --anneal 0.05 && python ssl_train.py --anneal 0.1 && python ssl_train.py --anneal 0.2"
pause