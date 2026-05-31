@echo off
REM ============================================================
REM 银行业研究 Agent - 每日自动任务入口
REM ============================================================
REM 用法:
REM   1) 直接双击,手动跑一次
REM   2) 用 Windows 任务计划程序定时跑(见同目录 SCHEDULED_RUN_SETUP.md)
REM ============================================================

cd /d D:\bank-sector-research-agent

echo.
echo [%date% %time%] === 任务开始 ===

REM ---- 激活虚拟环境 ----
call .venv\Scripts\activate.bat

REM ---- 1. 拉数据 ----
REM 社融通过 akshare 拉不到(SSL 问题),fetch_data 那一项会失败,但其他成功就行
echo.
echo [%date% %time%] [1/3] 拉取数据 ...
python src\fetch_data.py
REM 不 if errorlevel exit,因为单个指标失败不致命

REM ---- 2. 合并主表 ----
echo.
echo [%date% %time%] [2/3] 合并主表 ...
python src\clean_merge.py
if errorlevel 1 (
    echo [%date% %time%] [失败] clean_merge.py 出错
    pause
    exit /b 1
)

REM ---- 3. 生成宏观快报 ----
echo.
echo [%date% %time%] [3/3] 生成宏观快报 ...
python agent\macro_report.py
if errorlevel 1 (
    echo [%date% %time%] [失败] macro_report.py 出错
    pause
    exit /b 1
)

echo.
echo [%date% %time%] === 任务完成 ===
echo 最新快报在 outputs\ 目录,文件名 macro_flash_^<时间戳^>.txt
echo.

REM 取消下面这行的 REM 让脚本跑完后停住(手动跑时方便看)
REM 任务计划程序后台跑时建议保持 REM,跑完自动关闭
REM pause
