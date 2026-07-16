#!/usr/bin/env python3

import os
import glob
import pandas as pd

folder = "/home/frienkie/result"

# 查找所有包含data的xlsx
files = glob.glob(os.path.join(folder, "*data*.xlsx"))

if len(files) == 0:
    print("没有找到包含data的xlsx文件。")
    exit()

dfs = []

for file in files:
    print("读取:", os.path.basename(file))

    df = pd.read_excel(file)
    dfs.append(df)

# 合并所有数据
all_df = pd.concat(dfs, ignore_index=True)

# 列名
B = all_df.columns[1]
C = all_df.columns[2]
D = all_df.columns[3]
E = all_df.columns[4]
F = all_df.columns[5]

# -------------------------
# 总实验数、成功数
# -------------------------
count_df = (
    all_df.groupby(F)
    .agg(
        Total=(B, "count"),
        Success=(B, lambda x: (x == 0).sum())
    )
)

# -------------------------
# 仅成功的数据参与统计
# -------------------------
success_df = all_df[all_df[B] == 0]

# -------------------------
# 计算平均值和标准差
# -------------------------
stats = (
    success_df.groupby(F)
    .agg(
        distance_mean=(C, "mean"),
        distance_std=(C, "std"),

        time_mean=(D, "mean"),
        time_std=(D, "std"),

        velocity_mean=(E, "mean"),
        velocity_std=(E, "std")
    )
)

# NaN标准差（只有一个样本）设为0
stats = stats.fillna(0)

# -------------------------
# 保留小数
# -------------------------
stats["distance_mean"] = stats["distance_mean"].round(2)
stats["distance_std"] = stats["distance_std"].round(2)

stats["time_mean"] = stats["time_mean"].round(2)
stats["time_std"] = stats["time_std"].round(2)

stats["velocity_mean"] = stats["velocity_mean"].round(3)
stats["velocity_std"] = stats["velocity_std"].round(3)

# -------------------------
# 合成为 平均值±标准差
# -------------------------
stats["Distance"] = (
    stats["distance_mean"].map("{:.2f}".format)
    + "±"
    + stats["distance_std"].map("{:.2f}".format)
)

stats["Time"] = (
    stats["time_mean"].map("{:.2f}".format)
    + "±"
    + stats["time_std"].map("{:.2f}".format)
)

stats["Velocity"] = (
    stats["velocity_mean"].map("{:.3f}".format)
    + "±"
    + stats["velocity_std"].map("{:.3f}".format)
)

# 只保留需要输出的列
stats = stats[["Distance", "Time", "Velocity"]]

# -------------------------
# 合并
# -------------------------
result = count_df.join(stats)

# 成功率
result["Success_rate"] = result["Success"] / result["Total"]
result["Success_rate(%)"] = (result["Success_rate"] * 100).round(2)

# 恢复F列
result = result.reset_index()
result = result.rename(columns={F: "type"})

# 排序
result = result.sort_values("type")

# 保存
output = os.path.join(folder, "all_mean.xlsx")
result.to_excel(output, index=False)

print("完成！")
print("读取文件数：", len(files))
print("总数据行数：", len(all_df))
print("结果保存到：", output)
print("F为空:", all_df[F].isna().sum())