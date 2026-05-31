"""
Phase 3 extra: 命令行聊天模式

跟 macro_report.py 不同:macro_report 是「一次性生成报告」,
chat_mode 是「多轮对话」—— 启动时把主表关键数据塞到 system prompt,
然后你可以随便问任何关于银行/利率/社融的问题,模型基于真实数据回答。

支持的命令:
    /save [文件名]   保存当前对话到 outputs/(不填名字就用时间戳)
    /clear           清空对话历史(重新开始,system prompt 保留)
    /quit, /exit     退出
    Ctrl+C           强制退出

运行:
    python agent/chat_mode.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_PATH = PROJECT_ROOT / "data" / "master_monthly.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- LLM 配置 ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_ENABLED = True      # 聊天也开思考模式,质量好;觉得慢可改成 False
REASONING_EFFORT = "high"
MAX_TOKENS = 2000


def load_client():
    """加载 .env,创建 OpenAI 兼容的 DeepSeek 客户端。(同 macro_report)"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("错误:没有读到 DEEPSEEK_API_KEY")
        print(f"  请确认 {PROJECT_ROOT / '.env'} 存在,并已填入真实 key")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def build_initial_system():
    """读主表,把关键统计塞到 system prompt,让模型一上来就「有数据感」。

    返回一段 system 文字,内含:
      - 数据范围、字段说明
      - 最新一期各指标值
      - 历史分组规律(国债/社融 上行下行 vs 银行收益)
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["bank_return"] = df["bank_index"].pct_change() * 100

    latest = df.iloc[-1]

    # 算两个宏观变量的分组统计
    stats_lines = []
    for col, label in [("yield_10y", "10年国债收益率"),
                       ("shrzgm_yoy", "社融存量同比")]:
        sub = df[["bank_return", col]].copy()
        sub["change"] = sub[col].diff()
        sub = sub.dropna()
        up_mean = sub.loc[sub["change"] > 0, "bank_return"].mean()
        down_mean = sub.loc[sub["change"] < 0, "bank_return"].mean()
        up_n = int((sub["change"] > 0).sum())
        down_n = int((sub["change"] < 0).sum())
        stats_lines.append(
            f"  - {label}:上行月 {up_n} 个(银行均值 {up_mean:+.2f}%) | "
            f"下行月 {down_n} 个(银行均值 {down_mean:+.2f}%)"
        )

    return f"""你是一名资深卖方银行业首席分析师,基于以下真实数据回答用户问题。

【可用数据】2019-01 至 {latest['date'].date()},共 {len(df)} 个月度观测
字段:
  - bank_index:申万银行指数(月末收盘点位)
  - yield_10y:10 年期国债到期收益率(%)
  - shrzgm_yoy:社融存量同比(%)
  - lpr_1y, lpr_5y:LPR 1 年期 / 5 年期(%)

【最新一期】基准日 {latest['date'].date()}
  - 银行指数:{latest['bank_index']:.0f}
  - 10Y 国债:{latest['yield_10y']:.2f}%
  - 社融同比:{latest['shrzgm_yoy']:.2f}%
  - LPR 1Y / 5Y:{latest['lpr_1y']:.2f}% / {latest['lpr_5y']:.2f}%

【历史规律】(分组统计,2019-2025 月频)
{chr(10).join(stats_lines)}

回答风格:
- 用词准确、数据具体,避免「整体而言」「值得关注」等套话
- 必要时直接引用上面的数字
- 不知道的就说不知道,不要瞎编(尤其是数据范围外的内容)
- 涉及未来预测必须带不确定性表述
- 回答控制在 200-400 字,简洁有信息密度
"""


def call_llm(client, messages):
    """调用 LLM,带思考模式。返回回答文本。"""
    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    if THINKING_ENABLED:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = REASONING_EFFORT
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def save_conversation(messages, filename=None):
    """保存对话历史到 outputs/(跳过 system 那一条)。"""
    if filename is None:
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 确保扩展名
    if not filename.endswith(".txt"):
        filename += ".txt"
    path = OUTPUT_DIR / filename

    lines = [f"# 对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for m in messages[1:]:  # 跳过 system
        role_label = "你" if m["role"] == "user" else "助手"
        lines.append(f"## {role_label}")
        lines.append("")
        lines.append(m["content"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def chat_loop(client, system_prompt):
    """主对话循环。"""
    # messages 列表会一直累积:system + (user, assistant, user, assistant, ...)
    messages = [{"role": "system", "content": system_prompt}]

    print("\n" + "=" * 60)
    print(" 银行业研究助手 · 聊天模式 (DeepSeek-V4-Pro 思考模式)")
    print("=" * 60)
    print(" 提示:")
    print("   - 输入问题开始对话(可以问任何关于宏观/银行的问题)")
    print("   - /save [文件名]  保存对话")
    print("   - /clear          清空历史重新开始")
    print("   - /quit 或 /exit  退出")
    print("=" * 60 + "\n")

    while True:
        # ---- 读用户输入 ----
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            return

        if not user_input:
            continue

        # ---- 处理特殊命令 ----
        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)
            if cmd[0] in ("/quit", "/exit"):
                print("再见!")
                return
            if cmd[0] == "/clear":
                messages = [messages[0]]  # 只保留 system
                print("(对话历史已清空,system prompt 保留)\n")
                continue
            if cmd[0] == "/save":
                fname = cmd[1] if len(cmd) > 1 else None
                if len(messages) <= 1:
                    print("(还没有对话内容可以保存)\n")
                    continue
                path = save_conversation(messages, fname)
                print(f"(已保存到 {path})\n")
                continue
            print(f"(未知命令 {cmd[0]},可用:/save /clear /quit /exit)\n")
            continue

        # ---- 普通问答:加入历史 → 调 API → 加入回答 ----
        messages.append({"role": "user", "content": user_input})

        print("(思考中,请稍等 10-30 秒...)")
        try:
            reply = call_llm(client, messages)
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")
            # API 调用失败,把刚才的 user 消息回滚,免得污染历史
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})

        print(f"\n助手 > {reply}\n")


def main():
    print("初始化中...")
    client = load_client()
    system_prompt = build_initial_system()
    chat_loop(client, system_prompt)


if __name__ == "__main__":
    main()
