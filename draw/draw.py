import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==============================
# 1. 基本配置
# ==============================
DATA_PATH = "train.csv"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


# ==============================
# 2. 读取与整理数据
# ==============================
df = pd.read_csv(DATA_PATH)
df["dteday"] = pd.to_datetime(df["dteday"])

continuous_vars = [
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

discrete_vars = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit"
]

label_map = {
    "season": "季节",
    "yr": "年份",
    "mnth": "月份",
    "hr": "小时",
    "holiday": "是否节假日",
    "weekday": "星期",
    "workingday": "是否工作日",
    "weathersit": "天气状况",
    "temp": "温度",
    "atemp": "体感温度",
    "hum": "湿度",
    "windspeed": "风速",
    "cnt": "租车数量"
}


# ==============================
# 3. cnt 的频次分布
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, wspace=0.22)

sns.histplot(
    data=df,
    x="cnt",
    bins=40,
    kde=True,
    edgecolor="white",
    linewidth=0.7,
    ax=axes[0]
)

axes[0].axvline(
    df["cnt"].mean(),
    linestyle="--",
    linewidth=1.8,
    label=f"均值：{df['cnt'].mean():.1f}"
)
axes[0].axvline(
    df["cnt"].median(),
    linestyle=":",
    linewidth=2,
    label=f"中位数：{df['cnt'].median():.1f}"
)

axes[0].set_title("cnt 频次分布", fontsize=14, fontweight="bold", pad=12)
axes[0].set_xlabel("租车数量 cnt", fontsize=11)
axes[0].set_ylabel("样本频次", fontsize=11)
axes[0].legend(fontsize=10)

sns.histplot(
    np.log1p(df["cnt"]),
    bins=40,
    kde=True,
    edgecolor="white",
    linewidth=0.7,
    ax=axes[1]
)

axes[1].set_title("log(1 + cnt) 分布", fontsize=14, fontweight="bold", pad=12)
axes[1].set_xlabel("log(1 + cnt)", fontsize=11)
axes[1].set_ylabel("样本频次", fontsize=11)

fig.suptitle("目标变量 cnt 的分布特征", fontsize=18, fontweight="bold", y=0.96)

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_cnt_distribution.png"),
    bbox_inches="tight"
)
plt.show()


# ==============================
# 4. 连续变量与 cnt 的关系
# ==============================
n_cols = 2
n_rows = math.ceil(len(continuous_vars) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11))
axes = np.array(axes).reshape(-1)

# 重新设置间距，避免标题和坐标轴互相压住
fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.98, hspace=0.42, wspace=0.25)

for ax, col in zip(axes, continuous_vars):
    sns.regplot(
        data=df,
        x=col,
        y="cnt",
        lowess=True,
        scatter_kws={
            "alpha": 0.16,
            "s": 18,
            "edgecolor": "none"
        },
        line_kws={
            "linewidth": 2.5
        },
        ax=ax
    )

    corr = df[[col, "cnt"]].corr(method="spearman").iloc[0, 1]

    ax.set_title(
        f"{label_map[col]} 与 cnt\nSpearman 相关系数 = {corr:.3f}",
        fontsize=12.5,
        fontweight="bold",
        pad=10
    )
    ax.set_xlabel(label_map[col], fontsize=11, labelpad=6)
    ax.set_ylabel("cnt", fontsize=11, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)

fig.suptitle("连续变量与 cnt 的关系", fontsize=18, fontweight="bold", y=0.98)

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_continuous_vs_cnt.png"),
    bbox_inches="tight"
)
plt.show()


# ==============================
# 5. 离散变量与 cnt 的关系
# ==============================
n_cols = 2
n_rows = math.ceil(len(discrete_vars) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 22))
axes = np.array(axes).reshape(-1)

fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.98, hspace=0.52, wspace=0.22)

for ax, col in zip(axes, discrete_vars):
    sns.boxplot(
        data=df,
        x=col,
        y="cnt",
        showfliers=False,
        width=0.65,
        ax=ax
    )

    means = df.groupby(col, as_index=False)["cnt"].mean()

    ax.plot(
        range(len(means)),
        means["cnt"],
        marker="o",
        linewidth=2,
        label="组内均值"
    )

    ax.set_title(
        f"{label_map[col]} 与 cnt",
        fontsize=12.5,
        fontweight="bold",
        pad=10
    )
    ax.set_xlabel(label_map[col], fontsize=11, labelpad=6)
    ax.set_ylabel("cnt", fontsize=11, labelpad=6)
    ax.tick_params(axis="both", labelsize=10)

    # 对可能较密的刻度做旋转
    if col in ["mnth", "hr", "weekday"]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

    ax.legend(loc="upper right", fontsize=9)

fig.suptitle("离散变量不同取值下的 cnt 分布", fontsize=18, fontweight="bold", y=0.985)

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_discrete_vs_cnt.png"),
    bbox_inches="tight"
)
plt.show()


# ==============================
# 6. 日期与 cnt 的变化趋势
# ==============================
daily_cnt = df.groupby("dteday", as_index=False)["cnt"].sum()

plt.figure(figsize=(15, 5.8))
sns.lineplot(
    data=daily_cnt,
    x="dteday",
    y="cnt",
    linewidth=1.5
)

plt.title("每日 cnt 总量变化趋势", fontsize=16, fontweight="bold", pad=12)
plt.xlabel("日期", fontsize=11)
plt.ylabel("每日租车总量", fontsize=11)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_daily_cnt_trend.png"),
    bbox_inches="tight"
)
plt.show()

print(f"绘图完成，图片已保存到：{os.path.abspath(OUTPUT_DIR)}")