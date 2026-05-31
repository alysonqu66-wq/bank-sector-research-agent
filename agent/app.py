"""
Phase 4: Streamlit Web UI

把 4 个 Agent 整合到一个网页里:
  - 宏观快报(macro_report)
  - 新闻摘要(news_summary)
  - 研报抽取(report_extract)
  - 聊天模式(chat_mode)

启动:
    streamlit run agent/app.py
浏览器会自动打开 http://localhost:8501

依赖:除 requirements.txt 已有的 streamlit,其余 LLM/数据逻辑全部
从 agent/ 下其他脚本复用,不重复实现。
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

# 让 app.py 能 import 同级的其他 agent 脚本(macro_report 等)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import macro_report
import news_summary
import report_extract
import chat_mode


# ===================== 页面设置 =====================
st.set_page_config(
    page_title="银行业研究助手",
    layout="wide",
)

st.title("银行业研究助手")
st.caption("基于 DeepSeek-V4-Pro 思考模式 + 宏观数据的 4 合 1 工具集")


# ===================== 共享:LLM 客户端(全会话共享一个) =====================
@st.cache_resource
def get_client():
    """加载共享的 OpenAI 兼容客户端(读 .env 里的 DEEPSEEK_API_KEY)。
    用 @st.cache_resource 确保整个会话只创建一次。
    """
    try:
        return macro_report.load_client()
    except SystemExit:
        # load_client 找不到 key 会 sys.exit;Streamlit 里改成抛异常
        raise RuntimeError(
            "DEEPSEEK_API_KEY 没读到。请确认项目根的 .env 文件存在且填了真实 key。"
        )


# ===================== 4 个标签页 =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "宏观快报",
    "新闻摘要",
    "研报抽取",
    "聊天模式",
])


# --------- Tab 1: 宏观快报 ---------
with tab1:
    st.header("宏观敏感性快报生成")
    st.markdown(
        "基于 `data/master_monthly.csv` 的最新数据,结合 2019-2025 月度分组规律,"
        "由 DeepSeek-V4-Pro(思考模式)生成 4 段结构化研究员判断:"
        "**形势 / 判断 / 重点 / 风险**。"
    )

    col1, col2 = st.columns(2)
    with col1:
        override_y10 = st.checkbox("手动指定 10Y 国债收益率", value=False, key="o_y10")
        y10_val = None
        if override_y10:
            y10_val = st.number_input("10Y 国债收益率 (%)",
                                      value=1.85, step=0.01, format="%.2f",
                                      key="i_y10")
    with col2:
        override_shrz = st.checkbox("手动指定社融存量同比", value=False, key="o_shrz")
        shrz_val = None
        if override_shrz:
            shrz_val = st.number_input("社融存量同比 (%)",
                                       value=8.30, step=0.1, format="%.2f",
                                       key="i_shrz")

    if st.button("生成宏观快报", key="btn_macro", type="primary"):
        try:
            with st.spinner("读取主表并计算历史规律..."):
                df, stats = macro_report.compute_historical_stats()

            mock_args = SimpleNamespace(y10=y10_val, shrz=shrz_val)
            current = macro_report.get_current_values(df, mock_args)

            # 展示当前环境的关键数字
            st.subheader("当前环境")
            c1, c2, c3 = st.columns(3)
            c1.metric("10Y 国债",
                      f"{current['y10_curr']:.2f}%",
                      f"{current['y10_change_bp']:+.0f} bp")
            c2.metric("社融同比",
                      f"{current['shrz_curr']:.2f}%",
                      f"{current['shrz_change_pp']:+.2f} pp")
            c3.metric("近 3 月银行累计", f"{current['recent_3m']:+.2f}%")

            user_prompt = macro_report.build_prompt(current, stats)

            with st.spinner(f"调用 {macro_report.MODEL}(思考模式)生成中... 约 10-30 秒"):
                client = get_client()
                report = macro_report.call_llm(
                    client, macro_report.SYSTEM_PROMPT, user_prompt
                )

            st.subheader("快报正文")
            st.markdown(report)

        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# --------- Tab 2: 新闻摘要 ---------
with tab2:
    st.header("政策新闻摘要 + 银行业影响判断")
    st.markdown(
        "把任意政策/财经新闻粘贴进去,生成 5 段结构化摘要:"
        "**核心 / 政策类型 / 关键数字 / 对银行业影响 / 最受影响子板块**。"
    )

    news_text = st.text_area(
        "新闻文本",
        height=200,
        placeholder="例:中国人民银行决定,自2026年6月15日起下调金融机构存款准备金率0.5个百分点...",
        key="ta_news",
    )

    if st.button("生成摘要", key="btn_news", type="primary"):
        if not news_text.strip():
            st.warning("请先粘贴一段新闻")
        else:
            try:
                user_prompt = news_summary.build_prompt(news_text)
                with st.spinner(f"调用 {news_summary.MODEL}(思考模式)... 约 10-30 秒"):
                    client = get_client()
                    report = news_summary.call_llm(
                        client, news_summary.SYSTEM_PROMPT, user_prompt
                    )

                st.subheader("摘要正文")
                st.markdown(report)

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")


# --------- Tab 3: 研报抽取 ---------
with tab3:
    st.header("研报观点抽取(PDF)")
    st.markdown(
        "上传一份研报 PDF,自动抽取标题、目标价、评级、关键假设、风险点等结构化信息。"
        "扫描版 PDF 暂不支持(需要 OCR)。"
    )

    uploaded = st.file_uploader("选择研报 PDF", type=["pdf"], key="up_pdf")

    if uploaded is not None:
        st.success(f"已上传:{uploaded.name}({uploaded.size / 1024:.1f} KB)")

        if st.button("抽取观点", key="btn_extract", type="primary"):
            tmp_path = None
            try:
                # 把上传的 PDF 写到临时文件,因为 extract_pdf_text 接受路径
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                with st.spinner("解析 PDF 文本..."):
                    pdf_text, num_pages = report_extract.extract_pdf_text(tmp_path)

                st.info(f"页数 {num_pages} | 提取字符数 {len(pdf_text)}")

                user_prompt = report_extract.build_prompt(pdf_text)
                with st.spinner(
                    f"调用 {report_extract.MODEL}(思考模式)抽取... 约 20-60 秒"
                ):
                    client = get_client()
                    raw = report_extract.call_llm(
                        client, report_extract.SYSTEM_PROMPT, user_prompt
                    )

                data, err = report_extract.parse_json_safely(raw)

                if data is None:
                    st.error(f"JSON 解析失败:{err}")
                    with st.expander("模型原始返回"):
                        st.text(raw)
                else:
                    st.subheader("抽取结果(人读版)")
                    st.text(report_extract.render_readable(data))

                    with st.expander("JSON(机器读)"):
                        st.json(data)

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)


# --------- Tab 4: 聊天模式 ---------
with tab4:
    st.header("多轮对话")
    st.markdown(
        "和银行业研究助手对话。模型自带主表数据感:"
        "知道最新指标值、84 个月的历史规律,可以回答任意宏观/银行相关问题。"
    )

    # 初始化对话历史(只一次,放 session_state 跨 rerun 持久)
    if "chat_messages" not in st.session_state:
        try:
            with st.spinner("初始化对话上下文(读主表)..."):
                system_text = chat_mode.build_initial_system()
            st.session_state.chat_messages = [
                {"role": "system", "content": system_text}
            ]
        except Exception as e:
            st.error(f"初始化失败:{type(e).__name__}: {e}")
            st.session_state.chat_messages = [
                {"role": "system", "content": "你是一名银行业研究助手。"}
            ]

    # 工具栏:清空对话
    col_a, col_b = st.columns([5, 1])
    with col_b:
        if st.button("清空对话", key="btn_clear_chat"):
            # 只保留 system 那一条
            st.session_state.chat_messages = st.session_state.chat_messages[:1]
            st.rerun()

    # 渲染历史消息(跳过 system)
    for msg in st.session_state.chat_messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框(用户提交后追加并调用 LLM)
    if user_input := st.chat_input("问点关于银行/宏观的什么..."):
        # 1. 追加并渲染用户消息
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. 调 LLM,渲染助手消息
        with st.chat_message("assistant"):
            with st.spinner("思考中... 约 10-30 秒"):
                try:
                    client = get_client()
                    reply = chat_mode.call_llm(
                        client, st.session_state.chat_messages
                    )
                except Exception as e:
                    reply = f"[错误] {type(e).__name__}: {e}"
                    # 失败时回滚刚才的 user 消息,免得污染上下文
                    st.session_state.chat_messages.pop()
            st.markdown(reply)

        # 3. 把助手消息也追加进历史(供下一轮使用)
        if not reply.startswith("[错误]"):
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )


# ===================== 侧边栏:说明信息 =====================
st.sidebar.title("关于")
st.sidebar.markdown(
    """
**银行业研究助手 v1**

- 模型:DeepSeek-V4-Pro(思考模式 high)
- 数据:`data/master_monthly.csv`(84 个月)
- 配置:`.env` 文件读取 `DEEPSEEK_API_KEY`

**4 个 Agent**
- 宏观快报:基于历史规律生成研究员风格判断
- 新闻摘要:政策新闻 → 5 段结构化解读
- 研报抽取:PDF → JSON + 可读版
- 聊天模式:多轮对话,自带主表数据感
"""
)

st.sidebar.title("命令行入口")
st.sidebar.code(
    "python agent/macro_report.py\n"
    'python agent/news_summary.py --text "..."\n'
    "python agent/report_extract.py --pdf <路径>\n"
    "python agent/chat_mode.py",
    language="bash",
)
