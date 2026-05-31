"""
修正 iFinD 导出的社融存量同比文件格式。

iFinD 直接导出的 Excel(可能带 .csv 错误后缀)有两个小问题:
  1. 列名可能写错(例如 'data' 应为 'date')
  2. 末尾有「数据来源:同花顺iFinD」之类的说明行

这个脚本把它清洗成 clean_merge.py 能直接读的标准 CSV:
    date, shrzgm_yoy

输入:
    data/shrzgm_yoy.csv  (实际是 xlsx,iFinD 默认导出格式)
输出:
    data/shrzgm_yoy.csv  (覆盖原文件,变成真正的 CSV)

运行:
    python src/preprocess_shrzgm.py
"""

import os
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
FILE_PATH = os.path.join(DATA_DIR, "shrzgm_yoy.csv")


def main():
    if not os.path.exists(FILE_PATH):
        print(f"错误:找不到文件 {FILE_PATH}")
        return

    # iFinD 导出的本质是 xlsx,先用 read_excel 试;
    # 如果失败(说明已经是真 CSV,例如脚本第二次跑),退回 read_csv
    print(f"读取 {FILE_PATH} ...")
    try:
        df = pd.read_excel(FILE_PATH)
        print("  (按 Excel 格式读取成功)")
    except Exception:
        df = pd.read_csv(FILE_PATH)
        print("  (按 CSV 格式读取成功 - 可能脚本之前已跑过)")

    print(f"  原始列名: {list(df.columns)}")
    print(f"  原始行数: {len(df)}")

    # 把可能写错的列名统一改对
    # 'data' 是用户笔误,'日期'/'时间' 是中文导出可能,统一映射到 'date'
    # 含 'yoy' 或 '同比' 的列映射到 'shrzgm_yoy'
    rename_map = {}
    for c in df.columns:
        c_lower = str(c).lower()
        if c_lower in ("data", "日期", "时间", "date"):
            rename_map[c] = "date"
        elif "yoy" in c_lower or "同比" in str(c):
            rename_map[c] = "shrzgm_yoy"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  列名修正: {rename_map}")

    # 确认两列都在
    if "date" not in df.columns or "shrzgm_yoy" not in df.columns:
        print(f"错误:修正后仍缺列。当前列: {list(df.columns)}")
        return

    # 只保留这两列
    df = df[["date", "shrzgm_yoy"]].copy()

    # 转类型;errors='coerce' 会把无法解析的值变成 NaN
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["shrzgm_yoy"] = pd.to_numeric(df["shrzgm_yoy"], errors="coerce")

    # 删掉空行(iFinD 末尾的「数据来源」之类的说明行就是这一步被干掉的)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if before != len(df):
        print(f"  删掉 {before - len(df)} 行非数据行(通常是末尾的来源说明)")

    # 日期统一对齐到当月最后一天(月末)
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)

    # 排序
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n清洗后:")
    print(f"  行数:     {len(df)}")
    print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  数值范围: {df['shrzgm_yoy'].min():.2f}% ~ {df['shrzgm_yoy'].max():.2f}%")

    # 写回真正的 CSV(覆盖原文件)
    df.to_csv(FILE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n已保存为真正的 CSV: {FILE_PATH}")

    print("\n前 3 行:")
    print(df.head(3).to_string())
    print("\n后 3 行:")
    print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
