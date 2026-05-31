"""
Phase 3 功能 3: 研报观点抽取器

输入一份银行业研报 PDF,提取结构化的研究观点,
同时输出 JSON(机器可读)和 TXT(人可读)两份。

抽取字段:
    - report_meta:      研报元信息(标题、作者、机构、日期 等,能识别多少抽多少)
    - core_view:        核心观点(50-150 字)
    - target_price:     目标价(如果有,否则 null)
    - rating:           推荐评级(买入/增持/中性/减持/卖出/未提及)
    - key_assumptions:  关键假设(3-5 条)
    - risks:            主要风险点(3-5 条)
    - tags:             标签(子板块/主题/事件,3-5 个)

输入:
    必须给 --pdf 指向一份 PDF 文件

输出(在 outputs/):
    - report_extract_<PDF文件名>_<时间戳>.json   结构化数据
    - report_extract_<PDF文件名>_<时间戳>.txt    人读版

运行示例:
    python agent/report_extract.py --pdf data/some_bank_report.pdf
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv


# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- LLM 配置 ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_ENABLED = True
REASONING_EFFORT = "high"
MAX_TOKENS = 3000

# PDF 文本最大长度(字符数);超过会截断
# DeepSeek-V4-Pro context 较大,但研报有时几十页很长,需要控制
MAX_PDF_CHARS = 30000


# ---------- Prompt 组件 ----------
SYSTEM_PROMPT = """你是一名资深卖方银行业首席分析师,常年阅读同行研报。
你的任务:把一份研报的核心信息抽成结构化字段,供研究员快速比对多份研报观点。

抽取原则:
- 忠于原文,不要凭空编造(尤其是数字、评级、目标价)
- 如果某个字段在研报里没有明确给出,该字段值用 null(不是空字符串)
- 风险点要原文意思,不要套话(避免「市场风险」「政策风险」这种谁都能写的)
- 关键假设要具体(例如「假设 2026 年 LPR 再降 20bp」),不要「假设利率下行」这种模糊话
"""


def load_client():
    """同 macro_report,加载 .env + 创建 DeepSeek 客户端。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("错误:没有读到 DEEPSEEK_API_KEY")
        print(f"  请确认 {PROJECT_ROOT / '.env'} 存在,并已填入真实 key")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def extract_pdf_text(pdf_path):
    """用 pypdf 提取 PDF 全文。返回(文本, 页数)。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("错误:pypdf 没装,运行 pip install pypdf")
        sys.exit(1)

    p = Path(pdf_path)
    if not p.exists():
        print(f"错误:找不到 PDF 文件 {p}")
        sys.exit(1)
    if p.suffix.lower() != ".pdf":
        print(f"警告:{p} 不是 .pdf 后缀,继续尝试...")

    print(f"  读取 PDF: {p}")
    reader = PdfReader(str(p))
    num_pages = len(reader.pages)
    print(f"  页数:{num_pages}")

    # 逐页提取文本,中间加分隔符方便模型理解结构
    chunks = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"    [P{i}] 解析失败({type(e).__name__}),跳过")
            continue
        if text.strip():
            chunks.append(f"--- 第 {i} 页 ---\n{text}")

    full_text = "\n\n".join(chunks)
    print(f"  提取字符数:{len(full_text)}")

    # 截断
    if len(full_text) > MAX_PDF_CHARS:
        print(f"  文本超过 {MAX_PDF_CHARS} 字,截断到前 {MAX_PDF_CHARS} 字")
        full_text = full_text[:MAX_PDF_CHARS] + "\n\n[...内容被截断...]"

    return full_text, num_pages


def build_prompt(pdf_text):
    """构造抽取指令。要求模型严格按 JSON 格式输出。"""
    return f"""任务:从下面这份银行业研报里抽取结构化观点。

【严格输出格式】只输出**有效的 JSON**(以 `{{` 开头、以 `}}` 结尾,不要任何 markdown 代码块、不要解释、不要前后文字)。

JSON 字段定义:
{{
  "report_meta": {{
    "title": "研报标题(如能识别)",
    "author": "分析师姓名(如能识别)",
    "institution": "发布机构(如能识别,例如「中信证券」)",
    "report_date": "研报日期 YYYY-MM-DD(如能识别)",
    "covered_target": "覆盖的标的(银行名或子板块)"
  }},
  "core_view": "核心观点,50-150 字,要具体",
  "target_price": "目标价字符串如「7.50元」或 null(没明确给则 null)",
  "rating": "推荐评级:买入/增持/中性/减持/卖出 之一,或 null",
  "key_assumptions": [
    "关键假设1(具体,带数字)",
    "关键假设2",
    "..."
  ],
  "risks": [
    "风险点1(原文意思,不要套话)",
    "风险点2",
    "..."
  ],
  "tags": ["子板块或主题标签", "3-5 个"]
}}

【研报全文】
{pdf_text}

【再次强调】只输出 JSON,不要 ```json 包装,不要任何解释文字。
"""


def call_llm(client, system_prompt, user_prompt):
    """调用 DeepSeek,带思考 + 错误处理。"""
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
        sys.exit(1)


def parse_json_safely(text):
    """尝试把模型回答解析成 JSON。失败时,尝试剥掉 markdown 代码块再试。"""
    # 直接 parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    # 模型有时会用 ```json ... ``` 包,剥掉再 parse
    stripped = text.strip()
    if stripped.startswith("```"):
        # 去掉第一行 ```json 和最后一行 ```
        lines = stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        try:
            return json.loads("\n".join(lines)), None
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "顶层不是 JSON,且没有 ```json 包装"


def render_readable(data):
    """把解析后的 JSON 渲染成人可读的多段文字。"""
    meta = data.get("report_meta", {}) or {}
    lines = []
    lines.append("【研报元信息】")
    lines.append(f"  标题:    {meta.get('title') or '(未识别)'}")
    lines.append(f"  分析师:  {meta.get('author') or '(未识别)'}")
    lines.append(f"  机构:    {meta.get('institution') or '(未识别)'}")
    lines.append(f"  日期:    {meta.get('report_date') or '(未识别)'}")
    lines.append(f"  覆盖标的:{meta.get('covered_target') or '(未识别)'}")
    lines.append("")

    lines.append("【核心观点】")
    lines.append(f"  {data.get('core_view') or '(未抽取到)'}")
    lines.append("")

    tp = data.get("target_price")
    lines.append(f"【目标价】  {tp if tp else '(研报未给出)'}")
    rating = data.get("rating")
    lines.append(f"【评级】    {rating if rating else '(研报未给出)'}")
    lines.append("")

    lines.append("【关键假设】")
    for i, a in enumerate(data.get("key_assumptions", []) or [], 1):
        lines.append(f"  {i}. {a}")
    if not data.get("key_assumptions"):
        lines.append("  (未抽取到)")
    lines.append("")

    lines.append("【风险点】")
    for i, r in enumerate(data.get("risks", []) or [], 1):
        lines.append(f"  {i}. {r}")
    if not data.get("risks"):
        lines.append("  (未抽取到)")
    lines.append("")

    tags = data.get("tags", []) or []
    lines.append(f"【标签】    {' | '.join(tags) if tags else '(无)'}")
    return "\n".join(lines)


def save_outputs(data, raw_text, pdf_path, parse_error):
    """保存 JSON + TXT 两份。"""
    pdf_stem = Path(pdf_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / f"report_extract_{pdf_stem}_{ts}"

    if data is not None:
        # JSON 文件
        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # TXT 人读版
        txt_path = base.with_suffix(".txt")
        thinking_tag = f"思考模式 ON ({REASONING_EFFORT})" if THINKING_ENABLED else "思考模式 OFF"
        header = f"""研报观点抽取结果
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型:     {MODEL} | {thinking_tag}
PDF:      {pdf_path}

------------------------------------------------------------

"""
        txt_path.write_text(header + render_readable(data), encoding="utf-8")
        return json_path, txt_path

    # 解析失败的兜底:把原始返回存下来,方便人工排查
    fallback = base.with_suffix(".raw.txt")
    fallback.write_text(
        f"JSON 解析失败:{parse_error}\n\n--- 模型原始返回 ---\n{raw_text}",
        encoding="utf-8",
    )
    return None, fallback


def main():
    parser = argparse.ArgumentParser(description="研报观点抽取(PDF → JSON + TXT)")
    parser.add_argument("--pdf", type=str, required=True,
                        help="研报 PDF 文件路径(必填)")
    args = parser.parse_args()

    print("=" * 60)
    print(" 研报观点抽取器")
    print("=" * 60)

    # 1. 加载 LLM
    client = load_client()

    # 2. 提取 PDF 文本
    print("\n[1/3] 提取 PDF 文本 ...")
    pdf_text, num_pages = extract_pdf_text(args.pdf)
    if not pdf_text.strip():
        print("错误:PDF 没提取到任何文本(可能是扫描版/图片型 PDF,需要 OCR)")
        sys.exit(1)

    # 3. 调用 LLM
    thinking_tag = "思考模式 ON" if THINKING_ENABLED else "思考模式 OFF"
    print(f"\n[2/3] 调用 {MODEL} | {thinking_tag} | 抽取结构化观点 ...")
    print(f"      (PDF {num_pages} 页,思考模式 20—60 秒,请耐心等)")
    user_prompt = build_prompt(pdf_text)
    raw_response = call_llm(client, SYSTEM_PROMPT, user_prompt)

    # 4. 解析 JSON
    print(f"\n[3/3] 解析 JSON ...")
    data, err = parse_json_safely(raw_response)
    if data is None:
        print(f"  [失败] JSON 解析错误:{err}")
        print(f"         模型返回前 200 字:{raw_response[:200]}...")
    else:
        print(f"  [OK]   抽取到 {len(data)} 个顶层字段")

    # 5. 保存
    json_path, txt_path = save_outputs(data, raw_response, args.pdf, err)

    # 6. 打印结果
    if data is not None:
        print("\n" + "=" * 60)
        print(" 抽取结果(人读版)")
        print("=" * 60)
        print(render_readable(data))
        print("=" * 60)
        print(f"\nJSON: {json_path}")
        print(f"TXT:  {txt_path}")
    else:
        print(f"\n模型返回不是合法 JSON,原始内容已存到:{txt_path}")
        print(f"建议:重跑一次(模型偶尔会输出不规范),或把这个文件给我看看怎么改 prompt")


if __name__ == "__main__":
    main()
