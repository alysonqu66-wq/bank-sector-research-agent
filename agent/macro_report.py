"""
Phase 3 功能 2:宏观敏感性快报生成

读最新宏观数据(或手动指定),结合 Phase 1-2 算出的历史规律,
让 Claude API 生成一段约 150 字的「当前环境下银行业判断」。

输入逻辑:
  - 默认:从 data/master_monthly.csv 取最后一行作为「当前值」
  - 也可命令行手动指定当前 10Y 国债收益率和社融同比

输出:
  - 控制台打印
  - outputs/macro_flash_<时间戳>.txt(可用 --no-save 关闭)

运行示例:
    # 用主表最新值
    python agent/macro_report.py

    # 手动指定当前 10Y = 1.85, 社融同比 = 8.3
    python agent/macro_report.py --y10 1.85 --shrz 8.3

    # 只打印不保存
    python agent/macro_report.py --no-save
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ---------- 路径 ----------
# Path 用法和 os.path 等价,但更清晰;.parent 取上一级目录
HERE = Path(__file__).resolve().parent              # agent/
PROJECT_ROOT = HERE.parent                          # 项目根目录
DATA_PATH = PROJECT_ROOT / "data" / "master_monthly.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- 模型配置 ----------
# DeepSeek 用 OpenAI 兼容接口,只要设 base_url 和模型名就行,后续代码完全通用。
#
# 当前可用模型(2026 年):
#   "deepseek-v4-pro"   -- 旗舰模型,支持思考模式开关(推荐做研究/分析任务)
#   "deepseek-v4-flash" -- 轻量模型,便宜快但质量稍弱
#   "deepseek-chat"     -- 旧别名,2026-07-24 后停用(目前指向 v4-flash 非思考模式)
#   "deepseek-reasoner" -- 旧别名,2026-07-24 后停用(目前指向 v4-flash 思考模式)
#
# 想换其他 provider 只改 base_url + 模型名(例如 OpenAI:删 base_url,模型改 "gpt-4o-mini")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"    # 旗舰模型,做研究/分析最稳

# 思考模式开关:开启后模型会先输出一段「思维链」再给最终答案
#   True  = 质量更高但更慢 / 更贵(单次约多花 0.005-0.02 元)
#   False = 快/便宜,但对复杂判断类任务质量略弱
# 我们这是研究任务,默认开启
THINKING_ENABLED = True
REASONING_EFFORT = "high"    # "high" 或 "max",思考强度

MAX_TOKENS = 2000            # 结构化 4 段输出约 500—800 token,留余量


def load_client():
    """从 .env 读 API key,初始化 DeepSeek (OpenAI 兼容)客户端。"""
    # load_dotenv 会读项目根的 .env 文件,把里面的变量写到 os.environ
    # 这样后面 os.environ.get() 就能拿到了
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("错误:没有读到 DEEPSEEK_API_KEY")
        print(f"  请确认 {PROJECT_ROOT / '.env'} 存在,并已填入真实 key")
        print(f"  如果还没有 .env,运行:")
        print(f"    Copy-Item .env.example .env")
        print(f"    然后用记事本打开 .env 把 key 填进去")
        print(f"  注册和创建 key:https://platform.deepseek.com")
        sys.exit(1)

    # OpenAI SDK 支持自定义 base_url -- 这是兼容 DeepSeek/智谱/Ollama 等服务的标准做法
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def compute_historical_stats():
    """读主表,复算 Phase 2 的分组分析(让本脚本自包含,不依赖外部状态)。"""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 月度收益率
    df["bank_return"] = df["bank_index"].pct_change() * 100

    # 对两个核心宏观变量分别算「上行月 vs 下行月」的银行平均收益
    stats = {}
    for col, label in [("yield_10y", "10年国债收益率"),
                       ("shrzgm_yoy", "社融存量同比")]:
        sub = df[["bank_return", col]].copy()
        sub["change"] = sub[col].diff()
        sub = sub.dropna()

        stats[col] = {
            "label":     label,
            "up_mean":   sub.loc[sub["change"] > 0, "bank_return"].mean(),
            "down_mean": sub.loc[sub["change"] < 0, "bank_return"].mean(),
            "up_n":      int((sub["change"] > 0).sum()),
            "down_n":    int((sub["change"] < 0).sum()),
        }

    return df, stats


def get_current_values(df, args):
    """决定「当前」的宏观值,并计算多个上下文特征(分位、变化幅度、最近表现)。"""
    latest = df.iloc[-1]   # 最新月
    prev = df.iloc[-2]     # 上一月

    # 命令行指定就用命令行,否则用主表最新
    y10_curr = args.y10 if args.y10 is not None else float(latest["yield_10y"])
    shrz_curr = args.shrz if args.shrz is not None else float(latest["shrzgm_yoy"])

    y10_prev = float(prev["yield_10y"])
    shrz_prev = float(prev["shrzgm_yoy"])

    # 变化幅度
    # 国债收益率以 bp 计:0.01% = 1 bp,所以 pp 差 * 100 = bp
    y10_change_bp = (y10_curr - y10_prev) * 100
    # 社融同比直接用百分点(pp)
    shrz_change_pp = shrz_curr - shrz_prev

    # 历史分位:当前值在 5 年全样本中处于第几百分位(0 = 最低,100 = 最高)
    # 例如 1.85% 处于第 5 分位,说明历史上 95% 的月份比这个高
    y10_pct = (df["yield_10y"] < y10_curr).sum() / len(df) * 100
    shrz_pct = (df["shrzgm_yoy"] < shrz_curr).sum() / len(df) * 100

    # 最近 3 个月银行指数累计涨跌(用主表最后 4 行算)
    if len(df) >= 4:
        recent_3m = (df["bank_index"].iloc[-1] / df["bank_index"].iloc[-4] - 1) * 100
    else:
        recent_3m = float("nan")

    # 方向标签
    def direction(curr, prev):
        if curr > prev: return "上行"
        if curr < prev: return "下行"
        return "持平"

    return {
        "date":           str(latest["date"].date()),
        "y10_curr":       y10_curr,
        "y10_prev":       y10_prev,
        "y10_dir":        direction(y10_curr, y10_prev),
        "y10_change_bp":  y10_change_bp,
        "y10_pct":        y10_pct,
        "shrz_curr":      shrz_curr,
        "shrz_prev":      shrz_prev,
        "shrz_dir":       direction(shrz_curr, shrz_prev),
        "shrz_change_pp": shrz_change_pp,
        "shrz_pct":       shrz_pct,
        "recent_3m":      recent_3m,
    }


# ---------- Prompt 组件 ----------
# System message:给模型设人设,强约束写作风格
SYSTEM_PROMPT = """你是一名资深卖方银行业首席分析师,有 15 年从业经验,服务对象是公募/私募基金经理。

你的快报风格:
- 用词准确,数据具体,判断明确但带不确定性表述(例如「概率倾向」「预计」「需观察」)
- 拒绝复述已知信息,只输出基于数据的解读
- 涉及具体观点必须给出可操作落点(关注哪个事件 / 哪类标的 / 哪个数据)
- 拒绝套话(如「整体而言」「值得关注」等空话)
- 风险段要给出具体的反向情景,不要泛泛而谈
"""

# Few-shot 样例:让模型「看到」好的输出长什么样
FEW_SHOT_EXAMPLE = """【示例输入】
当前 10Y 国债 2.21%(下行 8bp,5 年第 18 分位)
当前社融同比 8.1%(下行 0.5pp,5 年第 5 分位)
最近 3 月银行累计 +5.2%

【示例输出】
【形势】10Y 国债加速下行至 2.21%(5 年 18 分位),社融同比跌至 8.1% 创 5 年新低,「弱信用 + 低利率」格局深化。
【判断】历史规律下两个方向均偏负面(国债下行月银行 -0.6%、社融下行月 +0.76%),但近 3 月银行已涨 5.2%,短期动能减弱,1 个月内偏震荡概率较高。
【重点】关注本月 LPR 报价(是否非对称下调压制净息差),以及大行中报息差指引;高股息防御组合相对占优。
【风险】若新一轮信贷数据进一步走弱,市场可能下修银行 ROE 假设,触发杀估值。"""


def build_prompt(current, stats):
    """构造给 LLM 的 user 消息。把历史规律、当前数据、上下文特征都塞进去。"""
    y10 = stats["yield_10y"]
    shrz = stats["shrzgm_yoy"]

    return f"""任务:基于下面的数据,生成一份银行业宏观敏感性快报。

【严格输出格式】中文,正好 4 段,每段独立成行,以 「【XX】」 开头,总长 200-250 字:
- 【形势】(~50 字):当前 10Y 国债与社融的水平、历史分位、本月变化幅度
- 【判断】(~60 字):基于历史规律的方向倾向,要带不确定性表述,不要给绝对结论
- 【重点】(~50 字):1-2 个具体应关注的事件、数据或细分标的(避免空话)
- 【风险】(~40 字):1 个具体的反向情景

【参考样例】仅供学习风格,不要照搬具体内容、数字、结论:
{FEW_SHOT_EXAMPLE}

---

【真正要分析的数据】

历史规律(2019-2025 月度数据,共 84 个月):
- 10Y 国债 *上行* 月({y10['up_n']} 个):银行平均月度收益 {y10['up_mean']:+.2f}%
- 10Y 国债 *下行* 月({y10['down_n']} 个):银行平均月度收益 {y10['down_mean']:+.2f}%
- 社融同比 *上行* 月({shrz['up_n']} 个):银行平均月度收益 {shrz['up_mean']:+.2f}%
- 社融同比 *下行* 月({shrz['down_n']} 个):银行平均月度收益 {shrz['down_mean']:+.2f}%

当前环境(基准日 {current['date']}):
- 10Y 国债收益率:{current['y10_curr']:.2f}% | 较上月 {current['y10_change_bp']:+.0f}bp({current['y10_dir']})| 5 年第 {current['y10_pct']:.0f} 分位
- 社融存量同比:{current['shrz_curr']:.2f}% | 较上月 {current['shrz_change_pp']:+.2f}pp({current['shrz_dir']})| 5 年第 {current['shrz_pct']:.0f} 分位
- 近 3 个月银行指数累计涨跌:{current['recent_3m']:+.2f}%

请按上面格式直接输出快报正文,无开头客套、无结尾总结。
"""


def call_llm(client, system_prompt, user_prompt):
    """调用 LLM API (DeepSeek,通过 OpenAI 兼容接口),带错误处理。"""
    # 准备调用参数,如果开了思考模式就多塞两个 DeepSeek 特有的参数
    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if THINKING_ENABLED:
        # extra_body 是 OpenAI SDK 提供的「逃生口」,
        # 用来传 OpenAI 标准外、provider 特有的参数(这里是 DeepSeek 的 thinking 配置)
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n错误:调用 LLM API 失败")
        print(f"  类型:{type(e).__name__}")
        print(f"  信息:{e}")
        print(f"  常见原因:1) API key 不对  2) 网络问题  3) 模型名 '{MODEL}' 不存在  4) 余额不足")
        print(f"  提示:模型名可在 https://api-docs.deepseek.com 查最新列表;")
        print(f"        如果 deepseek-v4-pro 失败,可临时换成 deepseek-v4-flash 或 deepseek-chat")
        sys.exit(1)


def save_output(report, current):
    """保存快报到 outputs/macro_flash_<时间戳>.txt"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"macro_flash_{ts}.txt"

    thinking_tag = f"思考模式 ON ({REASONING_EFFORT})" if THINKING_ENABLED else "思考模式 OFF"
    content = f"""宏观敏感性快报
基准日:     {current['date']}
生成时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型:       {MODEL} | {thinking_tag}

当前环境:
  - 10 年国债收益率: {current['y10_curr']:.2f}% | {current['y10_change_bp']:+.0f}bp | 5 年第 {current['y10_pct']:.0f} 分位
  - 社融存量同比:    {current['shrz_curr']:.2f}% | {current['shrz_change_pp']:+.2f}pp | 5 年第 {current['shrz_pct']:.0f} 分位
  - 近 3 月银行累计: {current['recent_3m']:+.2f}%

------------------------------------------------------------

{report}
"""
    path.write_text(content, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="基于历史规律 + 当前宏观数据,生成银行业判断快报"
    )
    parser.add_argument("--y10", type=float, default=None,
                        help="当前 10Y 国债收益率 (%%);不指定则用主表最新值")
    parser.add_argument("--shrz", type=float, default=None,
                        help="当前社融存量同比 (%%);不指定则用主表最新值")
    parser.add_argument("--no-save", action="store_true",
                        help="只打印,不保存到 txt")
    args = parser.parse_args()

    print("=" * 60)
    print(" 宏观敏感性快报生成器")
    print("=" * 60)

    # 1. 加载客户端
    client = load_client()

    # 2. 计算历史规律
    print("\n[1/3] 读主表并算历史规律 ...")
    df, stats = compute_historical_stats()
    print(f"  历史样本: {len(df)} 个月")
    for col in ["yield_10y", "shrzgm_yoy"]:
        s = stats[col]
        print(f"  {s['label']}: 上行{s['up_n']}月({s['up_mean']:+.2f}%) | "
              f"下行{s['down_n']}月({s['down_mean']:+.2f}%)")

    # 3. 当前环境(含分位、变化幅度、最近表现)
    current = get_current_values(df, args)
    print(f"\n[2/3] 当前环境(基准日 {current['date']}):")
    print(f"  10Y 国债: {current['y10_curr']:.2f}% | "
          f"{current['y10_change_bp']:+.0f}bp ({current['y10_dir']}) | "
          f"5 年第 {current['y10_pct']:.0f} 分位")
    print(f"  社融同比: {current['shrz_curr']:.2f}% | "
          f"{current['shrz_change_pp']:+.2f}pp ({current['shrz_dir']}) | "
          f"5 年第 {current['shrz_pct']:.0f} 分位")
    print(f"  近 3 个月银行指数累计: {current['recent_3m']:+.2f}%")

    # 4. 调用 LLM(传入 system + user 两段)
    thinking_tag = f"思考模式 ON({REASONING_EFFORT})" if THINKING_ENABLED else "思考模式 OFF"
    print(f"\n[3/3] 调用 {MODEL} | {thinking_tag} | 生成快报 ...")
    print(f"      (思考模式会更慢,可能要 10—30 秒,请耐心等)")
    user_prompt = build_prompt(current, stats)
    report = call_llm(client, SYSTEM_PROMPT, user_prompt)

    # 5. 打印
    print("\n" + "=" * 60)
    print(" 快报正文")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 6. 保存
    if not args.no_save:
        path = save_output(report, current)
        print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
