# 银行业宏观敏感性研究 · LLM 研究辅助工具集

> 用 Python 验证利率与信用环境对 A 股银行板块的传导规律,并基于 LLM API 搭建一组面向银行业研究员的辅助 Agent。
>
> **数据 + 统计分析 + LLM 工程化**三层完整闭环,从原始 CSV 一直跑到结构化研究输出。

---

## 项目简介

银行的盈利与股价对利率和信用环境高度敏感。本项目把这一研究问题拆成两层:

- **数据层(Phase 1—2)**:用 2019—2025 共 84 个月的月频数据,系统化验证 10 年国债收益率与社融存量同比对银行板块月度收益的影响
- **Agent 层(Phase 3)**:基于 DeepSeek-V4-Pro(OpenAI 兼容接口 + 思考模式)搭建 4 个独立可运行的 Agent

| Agent | 输入 | 输出 |
|---|---|---|
| 宏观敏感性快报 | 主表最新值 / 手动指定 | 结合历史规律的 4 段研究员风格判断 |
| 政策新闻摘要 | 新闻文本(粘贴或自动抓) | 5 段结构化摘要 + 银行业影响判断(利好/利空/中性) |
| 研报观点抽取 | 研报 PDF | JSON + TXT(核心观点、目标价、评级、关键假设、风险) |
| 命令行聊天模式 | 自由问答 | 自带主表数据感的多轮对话 |

此外提供一个 **Streamlit Web UI**(`agent/app.py`),把以上 4 个 Agent 整合到一个本地网页,
4 标签页 + 一个侧栏说明,适合演示与交互式探索。

---

## 核心发现(数据层结论)

在 2019—2025 的 84 个月观测中:

| 指标 | 结论 |
|---|---|
| 10Y 国债 *上行* 月份(38 个) | 银行平均月度收益 **+1.52%** |
| 10Y 国债 *下行* 月份(45 个) | 银行平均月度收益 **-0.60%** |
| 月度边际利率上行假说 | **支持**(差异 +2.12 个百分点) |
| 全样本 银行指数 vs 10Y 国债 相关系数 | -0.31(长期负相关) |
| 社融存量同比 *上行* 月份(36 个) | 银行平均月度收益 **-0.34%** |
| 社融存量同比 *下行* 月份(37 个) | 银行平均月度收益 **+0.76%** |
| 信用扩张利好银行假说 | **不支持**(反向证据,差异 -1.10 个百分点) |

> 「月度边际」(看月度变化方向)与「长期相关」(看水平时序)给出方向不同的信号,
> 同时社融对银行的影响呈反常识方向(可能反映「政策托底环境下市场担忧资产质量」)
> —— 这两个发现是 Agent 在生成研究输出时的核心数据基础。

---

## 技术栈

| 层 | 工具 |
|---|---|
| 数据获取 | [akshare](https://akshare.akfamily.xyz)(免费,无 API key);iFinD 手动导出兜底 |
| 数据处理 | pandas / numpy |
| 可视化 | matplotlib(SimHei 中文字体,300 dpi PNG) |
| LLM | OpenAI Python SDK + DeepSeek-V4-Pro(思考模式 high) |
| PDF 解析 | pypdf |
| 密钥管理 | python-dotenv + `.env`(已纳入 `.gitignore`) |
| 调度 | Windows 任务计划程序 + 一键启动 `.bat` |

---

## 目录结构

```
bank-sector-research-agent/
├── src/                            # 数据 ETL 与统计分析
│   ├── fetch_data.py               # akshare 拉原始数据
│   ├── preprocess_shrzgm.py        # iFinD 社融导出文件清洗
│   ├── clean_merge.py              # 合并为月频主表
│   └── analysis.py                 # 宏观敏感性分析 + 3 张核心图表
├── agent/                          # LLM 研究辅助工具
│   ├── macro_report.py             # 宏观敏感性快报生成
│   ├── news_summary.py             # 政策新闻摘要 + 银行业影响判断
│   ├── report_extract.py           # 研报 PDF → 结构化观点
│   ├── chat_mode.py                # 命令行多轮对话
│   └── app.py                      # Streamlit Web UI(4 标签页整合)
├── data/                           # 原始数据 + 月频主表(不入 git)
├── outputs/                        # 图表与生成报告(不入 git)
├── daily_run.bat                   # 一键自动化入口
├── SCHEDULED_RUN_SETUP.md          # Windows 定时任务配置指南
├── .env.example                    # 环境变量模板
├── requirements.txt
└── README.md
```

---

## 环境准备

```powershell
# 1. 创建虚拟环境(项目根目录)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 如遇 "在此系统上禁止运行脚本",先执行:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API key
Copy-Item .env.example .env
notepad .env   # 填入 DEEPSEEK_API_KEY(从 https://platform.deepseek.com 获取)
```

---

## 使用说明

### Phase 1:数据获取

```powershell
python src/fetch_data.py            # akshare 拉数据
python src/preprocess_shrzgm.py     # iFinD 导出的社融文件清洗(只在 akshare 拉不到时需要)
python src/clean_merge.py           # 合并为月频主表
```

产出 `data/master_monthly.csv`(84 行 × 6 列:`date, bank_index, yield_10y, shrzgm_yoy, lpr_1y, lpr_5y`)。

### Phase 2:数据分析

```powershell
python src/analysis.py
```

产出 3 张 300dpi 中文图表(`outputs/` 下):
- `01_bank_vs_yield_timeseries.png` — 银行指数 vs 10Y 国债双 Y 轴时序对比
- `02_yield_direction_groups.png` — 利率方向分组下的银行月度收益
- `03_credit_direction_groups.png` — 社融方向分组下的银行月度收益

### Phase 3:LLM Agent

```powershell
# 宏观敏感性快报
python agent/macro_report.py                       # 用主表最新值
python agent/macro_report.py --y10 1.85 --shrz 8.3 # 手动指定当前值

# 政策新闻摘要
python agent/news_summary.py --text "央行宣布..."     # 直接传文本
python agent/news_summary.py --file data/news_input.txt  # 从文件读
python agent/news_summary.py --auto 3                # 自动抓最近 3 天央视新闻

# 研报观点抽取
python agent/report_extract.py --pdf data/某研报.pdf

# 命令行聊天模式(多轮对话,自带主表数据感)
python agent/chat_mode.py

# Streamlit Web UI(把 4 个 Agent 整合到本地网页)
streamlit run agent/app.py
# 启动后浏览器自动打开 http://localhost:8501
```

所有 Agent 输出都会保存到 `outputs/` 下带时间戳的文件,便于追溯。

### 一键自动化

```powershell
daily_run.bat
```

按顺序跑 `fetch_data → clean_merge → macro_report` 全流程。
参见 `SCHEDULED_RUN_SETUP.md` 配置每天定时自动执行。

---

## 关键设计

| 设计点 | 实现 |
|---|---|
| **Provider 中立** | LLM 调用走 OpenAI 兼容 SDK,要切到 OpenAI / 智谱 / 通义 / Ollama 只改 `base_url` 和模型名 |
| **密钥安全** | `.env` + `python-dotenv`;`.gitignore` 严格排除;代码兜底检查模板值,避免占位符被当真值用 |
| **Prompt 工程三件套** | System 人设 + Few-shot 样例 + 强结构化输出格式约束 |
| **上下文特征工程** | 不只传原始数据,主动计算历史分位、变化幅度(bp/pp)、近期累计涨跌等研究员视角特征,让模型有真材料可解读 |
| **思考模式** | DeepSeek-V4-Pro 通过 `extra_body={"thinking": {"type": "enabled"}}` 开启,显著提升研究类任务质量 |
| **错误处理** | 每个 API 调用 try/except,报错信息直接给出排查方向(key/网络/模型名/余额) |
| **可追溯性** | 所有产出带时间戳归档;git 提交按 Phase 分阶段记录迭代 |

---

## 局限性

- 数据维度有限:目前仅覆盖银行指数、10Y 国债、社融、LPR 4 个核心指标,缺少 PMI/CPI/银行子板块/估值指标
- 社融数据需要从 iFinD 手动导出(akshare 该接口因数据源 SSL 配置问题不稳定)
- 研报抽取不支持扫描版 PDF(无 OCR 模块)
- 自动新闻抓取(`--auto`)依赖 akshare 的 `news_cctv` 接口,稳定性受上游影响
- 时间窗口起于 2019-01,未覆盖 2008 / 2015 等典型尾部事件,极端市场环境下结论外推需谨慎
- 主表为月频,无法捕捉日内或事件窗口(如政策公告当日)的短期效应

---

## 后续扩展方向

1. **扩指标**:加入 Shibor / R007 / M2 / 新增贷款 / PMI / CPI,以及银行子板块指数(国有大行 / 股份行 / 城商行 / 农商行)
2. **加事件维度**:整理降准 / 降息 / LPR 调整等政策事件标记表,让 Agent 能感知最近重大事件
3. **细化 prompt**:让快报输出区分子板块差异化判断
4. **OCR 接入**:支持扫描版研报 PDF
5. **延长样本**:接入更长历史数据,覆盖更完整的经济周期
6. **多模型对比**:支持一键切换 DeepSeek / OpenAI / 智谱 / Ollama 对相同 prompt 的输出做横向对比

---

## 依赖说明

完整依赖见 `requirements.txt`。核心:
- `akshare` ≥ 1.12
- `pandas` ≥ 2.0,`numpy` ≥ 1.24
- `matplotlib` ≥ 3.7
- `openai` ≥ 1.40(用于调用 DeepSeek 兼容接口)
- `python-dotenv`,`pypdf`,`openpyxl`,`feedparser`

Python 版本:3.11(其他 3.10+ 版本理论可用,未严格测试)。
