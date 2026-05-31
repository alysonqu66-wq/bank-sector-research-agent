"""
拉取银行业宏观敏感性研究所需的原始数据。

数据源:akshare(免费,无需 key)
时间范围:2019-01-01 至 2025-12-31

输出(写到 data/ 目录,各自一个 CSV):
  - sw_bank_801780.csv       申万银行行业指数(日频)
  - cn_10y_yield.csv         10年期中国国债到期收益率(日频)
  - shrzgm_yoy.csv           社会融资规模存量同比(月频)
  - lpr.csv                  LPR 1Y / 5Y(不规则变动)

运行:
  python src/fetch_data.py
"""

import os
import pandas as pd
import akshare as ak


# ---------- 配置 ----------
START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

# 把 data 路径定位到「本脚本所在目录的上一级 / data」
# 这样不管你在哪里启动 python,路径都是对的
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_sw_bank_index():
    """
    拉取申万银行行业指数(代码 801780)的日频收盘价。
    输出列:date, close
    """
    print("\n[1/4] 申万银行指数 801780 ...")
    try:
        # akshare 接口:index_hist_sw
        # 返回列通常含有:代码, 名称, 日期, 收盘, 开盘, 最高, 最低, 成交量, 成交额
        df = ak.index_hist_sw(symbol="801780", period="day")

        # 只保留日期和收盘价,并改成英文列名方便后续处理
        df = df[["日期", "收盘"]].copy()
        df.columns = ["date", "close"]

        # 日期统一转 datetime
        df["date"] = pd.to_datetime(df["date"])

        # 过滤时间范围
        mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
        df = df.loc[mask].sort_values("date").reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, "sw_bank_801780.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [OK]  保存 {out_path}")
        print(f"        {len(df)} 行 | {df['date'].min().date()} ~ {df['date'].max().date()}")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        print(f"  >>> 备用方案:用 iFinD 手动导出")
        print(f"      指标:申万银行(801780)日频收盘价")
        print(f"      时间:{START_DATE} 至 {END_DATE}")
        print(f"      存为:{DATA_DIR}\\sw_bank_801780.csv")
        print(f"      列名:date, close")
        return False


def fetch_cn_10y_yield():
    """
    拉取 10 年期中国国债到期收益率(日频)。
    输出列:date, yield_10y

    改用 bond_zh_us_rate 接口(返回中美各期限国债收益率),
    它比 bond_china_yield 更稳定,列名变化更小。
    """
    print("\n[2/4] 10年期中国国债到期收益率 ...")
    try:
        # 这个接口只接受 start_date,默认拉到今天
        df = ak.bond_zh_us_rate(start_date=START_DATE.replace("-", ""))

        # 返回的列名通常包含:日期、中国国债收益率2年/5年/10年/30年、美国国债收益率...
        # 用「找列」的方式定位 10 年那一列,这样以后列名稍有变动也不会挂
        col_10y = None
        for c in df.columns:
            if "中国" in c and "10年" in c:
                col_10y = c
                break
        if col_10y is None:
            raise ValueError(
                f"找不到中国10年国债列,实际列名:{list(df.columns)}"
            )

        # 找日期列(通常就是 '日期')
        date_col = "日期" if "日期" in df.columns else df.columns[0]

        df = df[[date_col, col_10y]].copy()
        df.columns = ["date", "yield_10y"]
        df["date"] = pd.to_datetime(df["date"])

        # 有些早期日期可能没有 10 年期数据,删掉这些缺失行
        df = df.dropna(subset=["yield_10y"])

        df = df.sort_values("date").reset_index(drop=True)

        # 过滤时间范围
        mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
        df = df.loc[mask].reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, "cn_10y_yield.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [OK]  保存 {out_path}")
        print(f"        {len(df)} 行 | {df['date'].min().date()} ~ {df['date'].max().date()}")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        print(f"  >>> 备用方案:用 iFinD 手动导出")
        print(f"      指标:10年期国债到期收益率(日频)")
        print(f"      时间:{START_DATE} 至 {END_DATE}")
        print(f"      存为:{DATA_DIR}\\cn_10y_yield.csv")
        print(f"      列名:date, yield_10y(单位:%)")
        return False


def fetch_shrzgm_yoy():
    """
    拉取社会融资规模存量,并计算同比(月频)。
    输出列:date, shrzgm_yoy(单位:%)

    说明:akshare 直接提供「存量」,同比 = (本月存量 / 上年同月存量 - 1) * 100,
    我们手动用 pct_change(12) 计算。
    """
    print("\n[3/4] 社会融资规模存量同比 ...")
    try:
        df = ak.macro_china_shrzgm()

        # 不同版本 akshare 返回列名可能略有差异,这里做一下兼容
        # 常见列名:'月份', '社会融资规模存量', '同比增量', '环比增量' 之类
        # 找到月份列与存量列
        month_col = None
        stock_col = None
        for c in df.columns:
            if "月份" in c or "month" in c.lower():
                month_col = c
            if "存量" in c and "同比" not in c and "环比" not in c:
                stock_col = c

        if month_col is None or stock_col is None:
            raise ValueError(
                f"无法在返回数据中识别月份列和存量列,实际列名:{list(df.columns)}"
            )

        # 月份格式通常是 'YYYYMM' 字符串
        df["date"] = pd.to_datetime(df[month_col].astype(str), format="%Y%m")
        df = df.sort_values("date").reset_index(drop=True)

        # 存量转 numeric,然后算同比
        df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")
        df["shrzgm_yoy"] = df[stock_col].pct_change(12) * 100

        # 只保留 date 和 yoy,过滤时间范围,并去掉前 12 个月的 NaN
        out = df[["date", "shrzgm_yoy"]].copy()
        mask = (out["date"] >= START_DATE) & (out["date"] <= END_DATE)
        out = out.loc[mask].dropna().reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, "shrzgm_yoy.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [OK]  保存 {out_path}")
        print(f"        {len(out)} 行 | {out['date'].min().date()} ~ {out['date'].max().date()}")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        print(f"  >>> 备用方案:用 iFinD 手动导出")
        print(f"      指标:社会融资规模存量同比(月频)")
        print(f"      时间:{START_DATE} 至 {END_DATE}")
        print(f"      存为:{DATA_DIR}\\shrzgm_yoy.csv")
        print(f"      列名:date, shrzgm_yoy(单位:%)")
        return False


def fetch_lpr():
    """
    拉取 LPR 1 年期与 5 年期历史利率(每次调整后才有一条新记录,非日频)。
    输出列:date, lpr_1y, lpr_5y(单位:%)
    """
    print("\n[4/4] LPR 1Y / 5Y ...")
    try:
        df = ak.macro_china_lpr()

        # 列名在不同版本可能是中文或英文,做一下兼容识别
        date_col = None
        col_1y = None
        col_5y = None
        for c in df.columns:
            c_upper = c.upper()
            if c in ("日期", "TRADE_DATE") or "日期" in c or "DATE" in c_upper:
                if date_col is None:
                    date_col = c
            if "1Y" in c_upper or "1年" in c:
                col_1y = c
            if "5Y" in c_upper or "5年" in c:
                col_5y = c

        if date_col is None or col_1y is None or col_5y is None:
            raise ValueError(
                f"无法识别 LPR 数据列,实际列名:{list(df.columns)}"
            )

        out = df[[date_col, col_1y, col_5y]].copy()
        out.columns = ["date", "lpr_1y", "lpr_5y"]
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)

        mask = (out["date"] >= START_DATE) & (out["date"] <= END_DATE)
        out = out.loc[mask].reset_index(drop=True)

        out_path = os.path.join(DATA_DIR, "lpr.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [OK]  保存 {out_path}")
        print(f"        {len(out)} 行 | {out['date'].min().date()} ~ {out['date'].max().date()}")
        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        print(f"  >>> 备用方案:用 iFinD 手动导出")
        print(f"      指标:LPR 1年期 和 5年期(每次调整即一条记录)")
        print(f"      时间:{START_DATE} 至 {END_DATE}")
        print(f"      存为:{DATA_DIR}\\lpr.csv")
        print(f"      列名:date, lpr_1y, lpr_5y(单位:%)")
        return False


def main():
    print("=" * 60)
    print(f" 数据获取范围: {START_DATE} 至 {END_DATE}")
    print(f" 保存目录:    {DATA_DIR}")
    print("=" * 60)

    results = [
        ("申万银行指数 801780", fetch_sw_bank_index()),
        ("10年期国债收益率",    fetch_cn_10y_yield()),
        ("社融存量同比",        fetch_shrzgm_yoy()),
        ("LPR 1Y / 5Y",         fetch_lpr()),
    ]

    print("\n" + "=" * 60)
    print(" 汇总:")
    for name, ok in results:
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {name}")
    print("=" * 60)

    if not all(ok for _, ok in results):
        print("\n注意:有数据未能从 akshare 拉取,请按上面提示用 iFinD 手动导出。")
        print("      手动导出后,直接放到 data/ 目录,然后继续运行 clean_merge.py 即可。")


if __name__ == "__main__":
    main()
