"""
Phase 3 功能 1: 政策新闻摘要 + 银行业影响判断

读一条政策/财经新闻文本,调用 DeepSeek-V4-Pro,
输出结构化摘要(5 段)+ 银行业影响判断(利好/利空/中性)。

输入(三选一,优先级:命令行 --text > --file > --auto > 默认文件):
    --text "..."           直接传一段新闻文本
    --file path/to/x.txt   从文件读
    --auto [天数]          用 akshare 拉最近 N 天央视新闻联播,
                           自动筛「银行/央行/LPR/降准/降息/金融」关键词
                           默认 3 天,例如 --auto 7 表示拉 7 天
    (不给参数)             默认读 data/news_input.txt

输出:
    - 控制台打印
    - outputs/news_summary_<时间戳>.txt(可用 --no-save 关闭)

运行示例:
    python agent/news_summary.py --text "央行5月20日宣布..."
    python agent/news_summary.py --file data/some_news.txt
    python agent/news_summary.py --auto 3
    python agent/news_summary.py            # 默认读 data/news_input.txt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from openai import OpenAI
from dotenv import load_dotenv


# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_INPUT = DATA_DIR / "news_input.txt"


# ---------- LLM 配置(同 macro_report,自包含) ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_ENABLED = True
REASONING_EFFORT = "high"
MAX_TOKENS = 2000


# ---------- Prompt 组件 ----------
SYSTEM_PROMPT = """你是一名资深卖方银行业首席分析师,15 年从业经验,
专门做政策新闻解读。你的工作场景:每天看到一条央行/监管新闻,要在 3 分钟内
告诉基金经理「这条新闻对哪些银行子板块利好/利空,传导链条是什么」。

你的输出风格:
- 用词精准,避免「整体而言」「值得关注」等空话
- 影响判断必须给具体的传导逻辑(经过哪些科目:净息差/资本充足率/不良率/估值)
- 不知道的就说不知道,不要硬编
- 短的新闻不要硬扩成长篇,核心说清楚即可
"""


def load_client():
    """加载 .env,创建 DeepSeek 客户端。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("错误:没有读到 DEEPSEEK_API_KEY")
        print(f"  请确认 {PROJECT_ROOT / '.env'} 存在,并已填入真实 key")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def read_news_from_file(path):
    """从文件读新闻,自动处理 UTF-8 / GBK 两种常见编码。"""
    p = Path(path)
    if not p.exists():
        print(f"错误:找不到新闻文件 {p}")
        print(f"  操作建议:")
        print(f"    1) 用记事本创建 {p},粘贴一段新闻文本,保存为 UTF-8")
        print(f"    2) 或运行:python agent/news_summary.py --text \"...\"")
        sys.exit(1)
    # 先试 UTF-8,失败回退 GBK(Windows 记事本旧版默认 GBK/ANSI)
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="gbk")
    text = text.strip()
    if not text:
        print(f"错误:文件 {p} 是空的,请先粘贴新闻内容")
        sys.exit(1)
    return text


def fetch_news_auto(days=3):
    """
    用 akshare 拉最近 N 天央视新闻联播,自动筛银行/金融相关关键词。
    返回筛中的第一条(最近的);如果一条都没筛中,返回 None。

    注意:akshare 接口偶尔会变/失效,失败就退回 None,让用户手动粘贴。
    """
    try:
        import akshare as ak
    except ImportError:
        print("[--auto] akshare 没装,跳过自动抓取")
        return None

    keywords = ["银行", "央行", "金融监管", "货币政策", "LPR", "降准", "降息",
                "存款准备金", "再贷款", "MLF", "公开市场操作"]

    print(f"[--auto] 用 akshare 拉最近 {days} 天央视新闻联播,筛选银行/金融关键词...")
    today = datetime.now()
    candidates = []

    for d in range(days):
        date_obj = today - timedelta(days=d)
        date_str = date_obj.strftime("%Y%m%d")
        try:
            df = ak.news_cctv(date=date_str)
        except Exception as e:
            print(f"  [{date_str}] 拉取失败({type(e).__name__}),跳过")
            continue
        if df is None or len(df) == 0:
            continue
        # akshare 返回的 df 通常有「标题」「内容」列
        for _, row in df.iterrows():
            text = " ".join(str(row.get(c, "")) for c in df.columns)
            if any(kw in text for kw in keywords):
                title = row.get("标题", "(无标题)")
                content = row.get("内容", "")
                candidates.append((date_str, str(title), str(content)))

    if not candidates:
        print(f"[--auto] 最近 {days} 天没筛到银行/金融相关新闻,请手动粘贴")
        return None

    # 取最近的(列表已按日期降序,因为 d 从 0 开始)
    date_str, title, content = candidates[0]
    print(f"[--auto] 匹配到 {len(candidates)} 条,使用最近的一条:")
    print(f"  日期:{date_str}")
    print(f"  标题:{title[:60]}{'...' if len(title) > 60 else ''}")
    return f"标题:{title}\n日期:{date_str}\n\n{content}"


def read_news(args):
    """根据命令行参数决定从哪儿读新闻。返回(新闻文本, 来源说明)。"""
    if args.text:
        return args.text, "命令行 --text"
    if args.auto is not None:
        text = fetch_news_auto(days=args.auto)
        if text is None:
            print("\n自动抓取没拿到,退回默认文件模式...\n")
            return read_news_from_file(DEFAULT_INPUT), f"默认文件 {DEFAULT_INPUT}(--auto 失败回退)"
        return text, f"akshare 自动抓取(最近 {args.auto} 天)"
    if args.file:
        return read_news_from_file(args.file), f"文件 {args.file}"
    return read_news_from_file(DEFAULT_INPUT), f"默认文件 {DEFAULT_INPUT}"


def build_prompt(news_text):
    """构造给 LLM 的 user 消息。"""
    return f"""任务:把下面这条新闻整理成「银行业研究员视角」的结构化摘要。

【严格输出格式】中文,正好 5 段,每段以「【XX】」开头,简洁不啰嗦:

【核心】一句话(<= 30 字)概括这条新闻在讲什么
【政策类型】从「货币政策 / 监管政策 / 财政政策 / 行业政策 / 其他」中选 1 个,后接 1 句简短说明
【关键数字】列出新闻中 1-3 个重要数字(利率/规模/比例 等);如果没有就写「无具体数字」
【对银行业影响】先给「利好 / 利空 / 中性」一个词,再给 60-100 字的传导逻辑,
                必须说明经过哪些维度:净息差 / 资本充足率 / 资产质量(不良) / 估值 / 流动性
【最受影响子板块】从「国有大行 / 股份行 / 城商行 / 农商行」中选 1-2 个,30 字内说明原因

---

【新闻原文】
{news_text}

---

请按上面格式直接输出,不要客套话、不要复述新闻原文。
"""


def call_llm(client, system_prompt, user_prompt):
    """调用 DeepSeek API,带思考模式 + 错误处理。"""
    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if THINKING_ENABLED:
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
        sys.exit(1)


def save_output(report, source_info, news_preview):
    """保存到 outputs/news_summary_<时间戳>.txt"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"news_summary_{ts}.txt"
    thinking_tag = f"思考模式 ON ({REASONING_EFFORT})" if THINKING_ENABLED else "思考模式 OFF"
    content = f"""政策新闻摘要 + 银行业影响判断
生成时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型:       {MODEL} | {thinking_tag}
输入来源:   {source_info}

【新闻原文(前 200 字预览)】
{news_preview}

------------------------------------------------------------

{report}
"""
    path.write_text(content, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="政策新闻摘要 + 银行业影响判断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", type=str, default=None,
                        help="直接传入一段新闻文本(适合短新闻)")
    parser.add_argument("--file", type=str, default=None,
                        help="从指定路径的文件读新闻")
    parser.add_argument("--auto", type=int, nargs="?", const=3, default=None,
                        help="用 akshare 拉最近 N 天央视新闻联播自动筛(默认 3 天)")
    parser.add_argument("--no-save", action="store_true",
                        help="只打印不保存到 txt")
    args = parser.parse_args()

    print("=" * 60)
    print(" 政策新闻摘要器 · 银行业影响判断")
    print("=" * 60)

    # 1. 加载 LLM 客户端
    client = load_client()

    # 2. 读取新闻
    print("\n[1/2] 读取新闻 ...")
    news_text, source = read_news(args)
    preview = news_text[:200] + ("..." if len(news_text) > 200 else "")
    print(f"  来源: {source}")
    print(f"  长度: {len(news_text)} 字")
    print(f"  预览: {preview[:80]}{'...' if len(preview) > 80 else ''}")

    # 3. 调用 LLM
    thinking_tag = "思考模式 ON" if THINKING_ENABLED else "思考模式 OFF"
    print(f"\n[2/2] 调用 {MODEL} | {thinking_tag} | 生成摘要 ...")
    print(f"      (思考模式 10—30 秒,请耐心等)")
    user_prompt = build_prompt(news_text)
    report = call_llm(client, SYSTEM_PROMPT, user_prompt)

    # 4. 打印
    print("\n" + "=" * 60)
    print(" 摘要正文")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 5. 保存
    if not args.no_save:
        path = save_output(report, source, preview)
        print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
