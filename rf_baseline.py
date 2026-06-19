"""
随机森林模型 - 共享单车租借量预测
精简特征工程 + 时序切分 + log(MSE) 优化 + 可视化
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 数据清洗 + 精简特征提取
# ============================================================
ONEHOT_COLS = ["season", "weathersit"]
CYCLIC_COLS = {"hr": 24, "weekday": 7, "mnth": 12}


def clean_and_extract(file_path):
    """读取 CSV，清洗 + 特征提取，返回 DataFrame"""
    df = pd.read_csv(file_path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # 湿度 0 -> 传感器故障（22 条），线性插值修复
    df["hum"] = (
        df["hum"].replace(0.0, np.nan)
        .interpolate(method="linear", limit_direction="both")
    )

    # 极端天气合并（仅 3 条）
    df["weathersit"] = df["weathersit"].replace(4, 3)

    # ---- 特征提取 ----
    date = df["dteday"]

    # 日期趋势
    df["day_of_year"] = date.dt.dayofyear
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)
    df["days_since_start"] = (date - pd.Timestamp("2011-01-01")).dt.days
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # 循环编码：替代 One-Hot，保留首尾相邻关系
    for col, period in CYCLIC_COLS.items():
        angle = 2 * np.pi * df[col] / period
        df[f"{col}_sin"] = np.sin(angle)
        df[f"{col}_cos"] = np.cos(angle)

    # 通勤高峰：树不易自动发现的强交互，显式构造
    df["is_morning_peak"] = (
        df["workingday"].eq(1) & df["hr"].between(7, 9)
    ).astype(int)
    df["is_evening_peak"] = (
        df["workingday"].eq(1) & df["hr"].between(16, 19)
    ).astype(int)

    return df


def prepare_train_test(train_path, test_path):
    """
    合并后 One-Hot 编码（仅 season/weathersit），确保列一致。
    返回 X_train, y_train, X_test, test_ids, feature_cols, train_dates
    """
    train = clean_and_extract(train_path)
    test = clean_and_extract(test_path)

    train["_src"] = "train"
    test["_src"] = "test"
    combined = pd.concat([train, test], ignore_index=True)

    combined = pd.get_dummies(
        combined, columns=ONEHOT_COLS, drop_first=False, dtype=int
    )

    train = combined[combined["_src"].eq("train")].copy()
    test = combined[combined["_src"].eq("test")].copy()
    train.drop(columns="_src", inplace=True)
    test.drop(columns="_src", inplace=True)

    # 保留日期用于时序切分
    train_dates = train["dteday"].copy()

    y_train = train.pop("cnt")
    test.drop(columns="cnt", inplace=True, errors="ignore")
    test_ids = test["ID"].copy()

    feature_cols = [c for c in train.columns if c not in ("ID", "dteday")]
    return train[feature_cols], y_train, test[feature_cols], test_ids, feature_cols, train_dates


# ============================================================
# 2. 数据准备
# ============================================================
print("数据加载与特征工程...")
X_train_all, y_train_all, X_test, test_ids, feature_cols, train_dates = prepare_train_test(
    "data-bike/train.csv", "data-bike/test.csv"
)
print(f"训练集: {X_train_all.shape}, 测试集: {X_test.shape}")
print(f"特征数: {len(feature_cols)}")

# ============================================================
# 3. 时序切分 + Log 变换
# ============================================================
# 数据已按时间排序，后 20% 作为验证集
split_idx = int(len(X_train_all) * 0.8)
print(f"\nTrain 时间范围: {train_dates.iloc[0].date()} ~ {train_dates.iloc[split_idx - 1].date()}")
print(f"Val   时间范围: {train_dates.iloc[split_idx].date()} ~ {train_dates.iloc[-1].date()}")
print(f"Test  时间范围: 2012-08-07 ~ 2012-12-31 (真实测试集)")

y_log = np.log1p(y_train_all.values)
print(f"\n原始 cnt 范围: [{y_train_all.min():.0f}, {y_train_all.max():.0f}]")
print(f"log(cnt+1) 范围: [{y_log.min():.3f}, {y_log.max():.3f}]")

# 时序切分（非随机！）
X_train = X_train_all.values[:split_idx]
X_val = X_train_all.values[split_idx:]
y_train_log = y_log[:split_idx]
y_val_log = y_log[split_idx:]
y_train_orig = np.expm1(y_train_log)
y_val_orig = np.expm1(y_val_log)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

# ============================================================
# 4. 训练
# ============================================================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=35,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
print("\n训练中（对数空间）...")
model.fit(X_train, y_train_log)

train_pred_log = model.predict(X_train)
val_pred_log = model.predict(X_val)
train_pred = np.expm1(train_pred_log)
val_pred = np.expm1(val_pred_log)

train_mse = mean_squared_error(y_train_orig, train_pred)
val_mse = mean_squared_error(y_val_orig, val_pred)
train_log_mse = mean_squared_error(y_train_log, train_pred_log)
val_log_mse = mean_squared_error(y_val_log, val_pred_log)

print(f"\n--- 原始空间 ---")
print(f"Train MSE: {train_mse:.2f}, RMSE: {np.sqrt(train_mse):.2f}")
print(f"Val   MSE: {val_mse:.2f}, RMSE: {np.sqrt(val_mse):.2f}")
print(f"\n--- 对数空间 (LogMSE) ---")
print(f"Train LogMSE: {train_log_mse:.4f}")
print(f"Val   LogMSE: {val_log_mse:.4f}")

# ============================================================
# 5. 特征重要性
# ============================================================
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\n特征重要性 Top 15:")
print(importances.head(15).to_string(index=False))

# ============================================================
# 6. 预测测试集
# ============================================================
test_pred_log = model.predict(X_test.values)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"ID": test_ids, "cnt": test_pred})
submission.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv 已保存, {len(submission)} 条")
print(f"预测 cnt 范围: [{test_pred.min():.1f}, {test_pred.max():.1f}]")

# ============================================================
# 7. 可视化
# ============================================================
print("生成可视化图表...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Random Forest with Time-Series Split — Training Visualization",
             fontsize=16, fontweight="bold")

# 图1: 特征重要性 Top 10
ax1 = axes[0, 0]
top10 = importances.head(10)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
ax1.barh(range(len(top10)), top10["importance"].values, color=colors, edgecolor="black")
ax1.set_yticks(range(len(top10)))
ax1.set_yticklabels(top10["feature"].values)
ax1.invert_yaxis()
ax1.set_xlabel("Importance")
ax1.set_title("Feature Importance (Top 10)")

# 图2: 预测值 vs 真实值
ax2 = axes[0, 1]
ax2.scatter(y_val_orig, val_pred, alpha=0.3, s=5, color="steelblue")
ax2.plot([y_val_orig.min(), y_val_orig.max()],
         [y_val_orig.min(), y_val_orig.max()], "r--", lw=2)
ax2.set_xlabel("True cnt")
ax2.set_ylabel("Predicted cnt")
ax2.set_title(f"Predicted vs True (Val)\nRMSE={np.sqrt(val_mse):.1f}")

# 图3: 残差分布
ax3 = axes[0, 2]
residuals = y_val_orig - val_pred
ax3.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
ax3.axvline(0, color="red", linestyle="--", lw=2)
ax3.set_xlabel("Residual (True - Predicted)")
ax3.set_ylabel("Frequency")
ax3.set_title(f"Residual Distribution (Val)\nMean={residuals.mean():.1f}, Std={residuals.std():.1f}")

# 图4: MSE 对比
ax4 = axes[1, 0]
x_labels = ["Train\n(original)", "Val\n(original)", "Train\n(log)", "Val\n(log)"]
x_vals = [train_mse, val_mse, train_log_mse, val_log_mse]
bar_colors = ["steelblue", "darkorange", "steelblue", "darkorange"]
bars = ax4.bar(x_labels, x_vals, color=bar_colors, edgecolor="black")
ax4.set_ylabel("MSE / LogMSE")
ax4.set_title("MSE Comparison (Original vs Log Space)")
for bar, val in zip(bars, x_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# 图5: 逐月 RMSE（验证集按月份分组）
ax5 = axes[1, 1]
val_dates = train_dates.iloc[split_idx:].reset_index(drop=True)
val_monthly = pd.DataFrame({"month": val_dates.dt.month})

monthly_rmse = {}
for m in sorted(val_monthly["month"].unique()):
    mask = val_monthly["month"] == m
    monthly_rmse[m] = np.sqrt(np.mean(
        (y_val_orig[mask.values] - val_pred[mask.values]) ** 2
    ))
months = list(monthly_rmse.keys())
rmse_vals = list(monthly_rmse.values())
bar_colors_m = ["darkorange" if m >= 8 else "steelblue" for m in months]
ax5.bar([str(m) + "月" for m in months], rmse_vals, color=bar_colors_m, edgecolor="black")
ax5.set_ylabel("RMSE")
ax5.set_title("Val RMSE by Month\n(orange = Test period months)")
ax5.axhline(np.sqrt(val_mse), color="red", linestyle="--", lw=1.5, label=f"Overall={np.sqrt(val_mse):.0f}")
ax5.legend(fontsize=8)

# 图6: cnt 分布 + 时序切分示意
ax6 = axes[1, 2]
ax6.hist(y_train_orig, bins=80, alpha=0.5, color="steelblue", label="Train cnt", density=True)
ax6.hist(y_val_orig, bins=80, alpha=0.5, color="darkorange", label="Val cnt", density=True)
ax6.set_xlabel("cnt")
ax6.set_ylabel("Density")
ax6.set_title("cnt Distribution: Train vs Val\n(Time-based split)")
ax6.legend()

plt.tight_layout()
plt.savefig("training_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("可视化图片已保存至 training_visualization.png")