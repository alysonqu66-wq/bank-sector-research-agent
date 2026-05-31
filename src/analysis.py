"""
Phase 2: 银行业宏观敏感性分析

基于 data/master_monthly.csv 完成 4 件事:
  1. 计算银行指数月度收益率
  2. 画核心图:银行指数 vs 10年国债收益率,双 Y 轴时序对比
  3. 把 10年国债收益率按"上行/下行"分组,看两种环境下银行月度收益率均值,
     验证"利率上行利好银行"的市场假设
  4. 对社融存量同比做同样的分组分析,验证"信用扩张利好银行"的假设

输出(全部存到 outputs/):
  01_bank_vs_yield_timeseries.png   时序对比图
  02_yield_direction_groups.png     利率方向 → 银行收益 柱状图
  03_credit_direction_groups.png    社融方向 → 银行收益 柱状图

运行:
  python src/analysis.py
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ---------- 中文字体设置(Windows) ----------
# SimHei 是 Windows 自带的"黑体",最稳定;Microsoft YaHei 备选
# axes.unicode_minus=False 让负号正常显示(不然会变成方框)
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------- 路径 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
OUTPUT_DIR = os.path.normpath(os.path.join(HERE, "..", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_PATH = os.path.join(DATA_DIR, "master_monthly.csv")
DPI = 300  # 高清晰度,适合放进报告或文档


def load_data():
    """读主表,并算银行指数月度收益率(单位:%)。"""
    df = pd.read_csv(MASTER_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 月度收益率 = (本月收盘 / 上月收盘 - 1) × 100
    # pct_change() 算的就是这个,乘 100 把小数变成百分点
    df["bank_return"] = df["bank_index"].pct_change() * 100

    print(f"加载主表:{len(df)} 行,{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  银行月度收益率统计:均值 {df['bank_return'].mean():+.2f}%,"
          f"标准差 {df['bank_return'].std():.2f}%")
    return df


def chart1_timeseries(df):
    """图 1:银行指数 vs 10年国债收益率,双 Y 轴时序图。"""
    print("\n[图 1] 银行指数 vs 10年国债收益率(时序对比)")

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # ---- 左 Y 轴:银行指数 ----
    color_left = "tab:blue"
    ax1.set_xlabel("日期", fontsize=11)
    ax1.set_ylabel("申万银行指数(点)", color=color_left, fontsize=11)
    ax1.plot(df["date"], df["bank_index"], color=color_left, linewidth=1.6, label="申万银行指数")
    ax1.tick_params(axis="y", labelcolor=color_left)
    ax1.grid(True, alpha=0.3)

    # ---- 右 Y 轴:10年国债到期收益率 ----
    ax2 = ax1.twinx()
    color_right = "tab:red"
    ax2.set_ylabel("10年期国债到期收益率(%)", color=color_right, fontsize=11)
    ax2.plot(df["date"], df["yield_10y"], color=color_right, linewidth=1.6, label="10年期国债收益率")
    ax2.tick_params(axis="y", labelcolor=color_right)

    plt.title("申万银行指数 vs 10年期国债收益率(2019—2025)", fontsize=14, pad=15)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "01_bank_vs_yield_timeseries.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    # 一句话结论:全样本相关系数
    corr = df[["bank_index", "yield_10y"]].corr().iloc[0, 1]
    if corr > 0.1:
        verdict = "弱-中等正相关 -- 利率上行期银行指数倾向偏强"
    elif corr < -0.1:
        verdict = "弱-中等负相关 -- 利率下行期银行指数倾向偏强"
    else:
        verdict = "几乎无相关 -- 银行指数与同期利率水平的线性关系不明显"

    print(f"  [保存] {out_path}")
    print(f"  [结论] 全样本相关系数 {corr:+.2f}: {verdict}")


def group_analysis(df, macro_col, macro_label, save_filename, hypothesis):
    """
    通用分组分析函数。
    把宏观变量按月度变化方向(上行 / 下行 / 持平)分组,
    比较每组中银行的平均月度收益率。
    """
    print(f"\n[分组分析] {macro_label}")

    # 计算宏观变量的月度变化(delta)
    sub = df[["date", "bank_return", macro_col]].copy()
    sub["macro_change"] = sub[macro_col].diff()

    # 去掉缺失行(第一行肯定 NaN,因为没有"上月")
    sub = sub.dropna(subset=["bank_return", "macro_change"]).reset_index(drop=True)

    # 三个组:上行 / 下行 / 持平
    up_mask = sub["macro_change"] > 0
    down_mask = sub["macro_change"] < 0
    flat_mask = sub["macro_change"] == 0

    up_mean = sub.loc[up_mask, "bank_return"].mean()
    down_mean = sub.loc[down_mask, "bank_return"].mean()
    up_n = int(up_mask.sum())
    down_n = int(down_mask.sum())
    flat_n = int(flat_mask.sum())

    print(f"  {macro_label} 上行月份({up_n} 个):银行平均月度收益 {up_mean:+.2f}%")
    print(f"  {macro_label} 下行月份({down_n} 个):银行平均月度收益 {down_mean:+.2f}%")
    if flat_n > 0:
        print(f"  {macro_label} 持平月份({flat_n} 个):不参与对比")

    diff = up_mean - down_mean
    support = "支持" if diff > 0 else "不支持(反向证据)"
    print(f"  [结论] 上行 - 下行 = {diff:+.2f} 百分点 -- {hypothesis}:{support}")

    # ---- 画柱状图 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = [f"{macro_label}上行\n(n={up_n})", f"{macro_label}下行\n(n={down_n})"]
    values = [up_mean, down_mean]
    # 大于 0 的柱用偏红色(银行赚钱),小于 0 的用偏灰色(银行亏)
    colors = ["#d62728" if v > 0 else "#7f7f7f" for v in values]
    bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor="black", width=0.5)

    # 在每根柱上方/下方标数值
    for bar, v in zip(bars, values):
        y_offset = 0.08 if v >= 0 else -0.25
        ax.text(bar.get_x() + bar.get_width() / 2, v + y_offset,
                f"{v:+.2f}%", ha="center", fontsize=12, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.6)
    ax.set_ylabel("银行指数平均月度收益率(%)", fontsize=11)
    ax.set_title(f"{macro_label}方向 → 银行月度收益(2019—2025)", fontsize=13, pad=12)
    ax.grid(True, alpha=0.3, axis="y")

    # 让 Y 轴上下都留点余白,标注不会贴边
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.3, y_max + 0.5)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [保存] {out_path}")


def main():
    print("=" * 60)
    print(" Phase 2: 银行业宏观敏感性分析")
    print("=" * 60)

    df = load_data()

    # 图 1: 双轴时序图
    chart1_timeseries(df)

    # 分组分析 1: 10年国债收益率方向
    group_analysis(
        df,
        macro_col="yield_10y",
        macro_label="10Y国债收益率",
        save_filename="02_yield_direction_groups.png",
        hypothesis="假设「利率上行利好银行(净息差扩张)」",
    )

    # 分组分析 2: 社融存量同比方向
    group_analysis(
        df,
        macro_col="shrzgm_yoy",
        macro_label="社融存量同比",
        save_filename="03_credit_direction_groups.png",
        hypothesis="假设「信用扩张(社融加速)利好银行」",
    )

    print("\n" + "=" * 60)
    print(" Phase 2 完成,3 张图已保存到 outputs/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
