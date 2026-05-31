"""
合并各数据源,生成月频主表 data/master_monthly.csv

逻辑:
  - 日频数据(银行指数、10年国债收益率)按月末取最后一个有效值
  - 月频数据(社融同比)直接对齐到月末
  - 不规则数据(LPR)按月末取最后一次公布的利率
  - 缺失值用前向填充(ffill)处理
  - 打印缺失统计和时间范围

运行:
  python src/clean_merge.py
"""

import os
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))


def load_csv_if_exists(filename):
    """
    安全读取 CSV:文件存在就读,不存在就返回 None 并打印提示。
    所有 CSV 都假设有 'date' 列。
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  [缺失] {filename}  (没找到这个文件)")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  [OK]   {filename}: {len(df)} 行 | "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def to_month_end(df, value_cols):
    """
    把数据按「月末」对齐:每月最后一个有效值。
    适用于日频(收益率、指数)和不规则数据(LPR)。
    """
    df = df.set_index("date").sort_index()
    # 'ME' 是月末频率(pandas 2.2 起把 'M' 改成 'ME');.last() 取每月最后一条非缺失记录
    df_m = df[value_cols].resample("ME").last()
    df_m = df_m.reset_index()
    return df_m


def main():
    print("=" * 60)
    print(" 读取各原始数据:")
    print("-" * 60)
    sw   = load_csv_if_exists("sw_bank_801780.csv")
    y10  = load_csv_if_exists("cn_10y_yield.csv")
    shrz = load_csv_if_exists("shrzgm_yoy.csv")
    lpr  = load_csv_if_exists("lpr.csv")

    # 至少要有一个数据,否则没法做合并
    if all(x is None for x in [sw, y10, shrz, lpr]):
        print("\n错误:没有任何数据文件,请先运行 src/fetch_data.py")
        return

    print("\n降采样到月频:")
    print("-" * 60)

    monthly_parts = []  # 用来装所有月频 DataFrame,最后合并

    if sw is not None:
        sw_m = to_month_end(sw, ["close"]).rename(columns={"close": "bank_index"})
        monthly_parts.append(sw_m)
        print(f"  银行指数(月末收盘):   {len(sw_m)} 行")

    if y10 is not None:
        y10_m = to_month_end(y10, ["yield_10y"])
        monthly_parts.append(y10_m)
        print(f"  10年国债收益率(月末): {len(y10_m)} 行")

    if shrz is not None:
        # 社融已经是月频,但日期可能在月初/月中,统一调整到当月月末
        shrz2 = shrz.copy()
        shrz2["date"] = shrz2["date"] + pd.offsets.MonthEnd(0)
        monthly_parts.append(shrz2)
        print(f"  社融存量同比(月末对齐):{len(shrz2)} 行")

    if lpr is not None:
        # LPR 是不规则数据(每月才公布一次),取每月最后一次报价
        lpr_m = to_month_end(lpr, ["lpr_1y", "lpr_5y"])
        monthly_parts.append(lpr_m)
        print(f"  LPR 1Y/5Y(月末):     {len(lpr_m)} 行")

    print("\n合并各数据源:")
    print("-" * 60)
    master = monthly_parts[0]
    for df in monthly_parts[1:]:
        master = pd.merge(master, df, on="date", how="outer")
    master = master.sort_values("date").reset_index(drop=True)
    print(f"  合并后行数: {len(master)}")
    print(f"  列:        {list(master.columns)}")

    # 缺失统计 - 填充前
    print("\n缺失值(前向填充前):")
    print("-" * 60)
    miss_before = master.isna().sum()
    for col, n in miss_before.items():
        print(f"  {col:20s}  缺失 {n} 个")

    # 前向填充:用上一个有效值填补缺失
    # 对 LPR、社融同比这种「公布即生效到下次公布」的数据是合理的
    master_filled = master.ffill()

    print("\n缺失值(前向填充后):")
    print("-" * 60)
    miss_after = master_filled.isna().sum()
    for col, n in miss_after.items():
        print(f"  {col:20s}  缺失 {n} 个")
    print("  (剩余的缺失通常出现在序列开头,因为前面没有数据可填充)")

    # 保存
    out_path = os.path.join(DATA_DIR, "master_monthly.csv")
    master_filled.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f" 主表已保存: {out_path}")
    print(f"   行数:       {len(master_filled)}")
    print(f"   时间范围:   {master_filled['date'].min().date()} ~ {master_filled['date'].max().date()}")
    print(f"   字段:       {list(master_filled.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
