"""
XGBoost 优化版 - 共享单车租借量预测

相较于上一版，本文件主要改动：
1. 保留已有清洗与特征工程：
   - 日期趋势特征
   - 小时/星期/月周期正余弦编码
   - 工作日早晚高峰与时间交互
   - 温度、湿度、风速非线性与组合特征
   - train/test 合并独热编码后再拆分，保证特征列一致

2. 针对上一版 Train RMSE 明显低于 Val RMSE 的问题，降低 XGBoost 复杂度：
   - max_depth 从 6 降到 4
   - min_child_weight 增大
   - subsample / colsample_bytree 降低
   - 增强 reg_alpha / reg_lambda 正则化

3. 保留时间顺序验证集，用于评估模型泛化能力。

4. 增加 final retrain：
   - 先用前 80% / 后 20% 验证确定模型效果和最佳树数量
   - 再用全部训练集重新训练最终模型
   - 用最终模型预测 test.csv
   - 生成 submission_xgboost_optimized.csv

运行前：
pip install xgboost

运行：
python xgboost_optimized.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("未安装 xgboost，请先运行：pip install xgboost") from exc


warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 全局配置
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data-bike"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_SUBMISSION = PROJECT_DIR / "submission_xgboost_optimized.csv"
OUTPUT_IMPORTANCE = PROJECT_DIR / "xgboost_optimized_feature_importance.csv"
OUTPUT_FIGURE = PROJECT_DIR / "xgboost_optimized_visualization.png"

RANDOM_STATE = 42

# time：更贴近真实“用历史预测未来”的场景
# random：方便和随机森林 baseline 做同划分对比
VALIDATION_MODE = "random"

# cnt 往往右偏，log1p 有时有帮助；建议先 False 跑一版，再改 True 对比
USE_LOG_TARGET = False

# 是否保留 windspeed=0。上一版把 windspeed=0 也当作缺失插值。
# 但 0 风速可能是真实无风，所以这里默认只修复 hum=0，保留 windspeed=0。
# 如果想复现上一版清洗，可改为 True。
INTERPOLATE_ZERO_WINDSPEED = False

CATEGORICAL_COLUMNS = ["season", "mnth", "hr", "weekday", "weathersit"]

# ============================================================
# 1. 清洗与特征提取
# ============================================================
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """围绕时间周期、通勤规律和天气影响提取特征。"""
    df = df.copy()
    date = pd.to_datetime(df["dteday"])

    # 日期位置与长期趋势特征
    df["day_of_month"] = date.dt.day
    df["day_of_year"] = date.dt.dayofyear
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)
    df["days_since_start"] = (date - pd.Timestamp("2011-01-01")).dt.days

    # 注意：原数据 weekday 通常 0 和 6 表示周末
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # 周期变量正余弦编码，避免 23 点与 0 点、12 月与 1 月被模型误认为距离很远
    for column, period in (("hr", 24), ("weekday", 7), ("mnth", 12)):
        angle = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)

    # 工作日通勤高峰及时间交互
    df["is_morning_peak"] = (
        df["workingday"].eq(1) & df["hr"].between(7, 9)
    ).astype(int)
    df["is_evening_peak"] = (
        df["workingday"].eq(1) & df["hr"].between(16, 19)
    ).astype(int)
    df["workingday_hour"] = df["workingday"] * df["hr"]
    df["weekend_hour"] = df["is_weekend"] * df["hr"]

    # 天气与体感相关的非线性 / 组合特征
    df["temp_feel_diff"] = df["temp"] - df["atemp"]
    df["temp_sq"] = df["temp"] ** 2
    df["hum_sq"] = df["hum"] ** 2
    df["windspeed_sq"] = df["windspeed"] ** 2
    df["temp_hum"] = df["temp"] * df["hum"]
    df["temp_windspeed"] = df["temp"] * df["windspeed"]
    df["hum_windspeed"] = df["hum"] * df["windspeed"]
    df["bad_weather"] = df["weathersit"].ge(3).astype(int)

    # 高峰 + 天气交互：坏天气对高峰需求可能影响更明显
    df["peak_bad_weather"] = (
        (df["is_morning_peak"] | df["is_evening_peak"]) & df["bad_weather"].eq(1)
    ).astype(int)

    df["dteday"] = date
    return df


def clean_and_extract(file_path: Path) -> pd.DataFrame:
    """读取原始 CSV，完成基础清洗与特征提取。"""
    df = pd.read_csv(file_path)
    df["dteday"] = pd.to_datetime(df["dteday"])

    # 按时间排序，让插值与时间切分更合理
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # hum=0 基本可视作异常/缺失
    if "hum" in df.columns:
        df["hum"] = (
            df["hum"]
            .replace(0.0, np.nan)
            .interpolate(method="linear", limit_direction="both")
        )

    # windspeed=0 可能是真实无风，也可能是缺失；默认保留，作为一个可对比实验开关
    if INTERPOLATE_ZERO_WINDSPEED and "windspeed" in df.columns:
        df["windspeed"] = (
            df["windspeed"]
            .replace(0.0, np.nan)
            .interpolate(method="linear", limit_direction="both")
        )

    # 极端天气 4 样本通常很少，合并到 3
    if "weathersit" in df.columns:
        df["weathersit"] = df["weathersit"].replace(4, 3)

    return extract_features(df)


def prepare_train_test(train_path: Path, test_path: Path):
    """
    生成 X, y, X_test, test_ids。
    train/test 合并后独热编码，再拆分，保证两边特征列完全一致。
    """
    train = clean_and_extract(train_path)
    test = clean_and_extract(test_path)

    train["_data_source"] = "train"
    test["_data_source"] = "test"

    combined = pd.concat([train, test], ignore_index=True)

    existing_categorical_cols = [
        col for col in CATEGORICAL_COLUMNS if col in combined.columns
    ]

    combined = pd.get_dummies(
        combined,
        columns=existing_categorical_cols,
        drop_first=False,
        dtype=int,
    )

    train_selected = combined[combined["_data_source"].eq("train")].copy()
    test_selected = combined[combined["_data_source"].eq("test")].copy()

    train_selected.drop(columns="_data_source", inplace=True)
    test_selected.drop(columns="_data_source", inplace=True)

    y = train_selected.pop("cnt")
    test_selected.drop(columns="cnt", inplace=True, errors="ignore")

    test_ids = test_selected["ID"].copy()

    X = train_selected.drop(columns=["ID", "dteday", "casual", "registered"], errors="ignore")
    X_test = test_selected.drop(columns=["ID", "dteday", "casual", "registered"], errors="ignore")

    # 再保险：确保 test 与 train 列完全一致
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    return X, y, X_test, test_ids


# ============================================================
# 2. 训练/验证集划分
# ============================================================
def split_train_valid(X: pd.DataFrame, y: pd.Series):
    if VALIDATION_MODE == "time":
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

    elif VALIDATION_MODE == "random":
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError("VALIDATION_MODE 只能是 'time' 或 'random'")

    return X_train, X_val, y_train, y_val


# ============================================================
# 3. XGBoost 模型配置
# ============================================================
def make_xgb_model(n_estimators: int = 5000, early_stopping_rounds: int | None = 150):
    """
    参数比上一版更保守，用于缓解过拟合。
    如果是 final retrain，没有验证集，则 early_stopping_rounds 传 None。
    """
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.03,

        # 降低树复杂度
        max_depth=4,
        min_child_weight=6,

        # 随机采样，降低过拟合
        subsample=0.8,
        colsample_bytree=0.8,

        # 正则化
        reg_alpha=0.1,
        reg_lambda=3.0,

        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds

    return XGBRegressor(**params)


def transform_target(y: pd.Series | np.ndarray):
    if USE_LOG_TARGET:
        return np.log1p(y)
    return y


def inverse_transform_pred(pred: np.ndarray):
    if USE_LOG_TARGET:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# ============================================================
# 4. 验证训练
# ============================================================
def train_validation_model(X_train, y_train, X_val, y_val):
    y_train_fit = transform_target(y_train)
    y_val_fit = transform_target(y_val)

    model = make_xgb_model(n_estimators=5000, early_stopping_rounds=150)

    print("开始训练 XGBoost 验证模型 ...")
    model.fit(
        X_train,
        y_train_fit,
        eval_set=[(X_val, y_val_fit)],
        verbose=100,
    )

    return model


def get_best_n_estimators(model) -> int:
    """
    读取 early stopping 得到的最佳树数量。
    不同 xgboost 版本属性略有差异，因此做兼容处理。
    """
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is not None:
        return int(best_iteration) + 1

    best_ntree_limit = getattr(model, "best_ntree_limit", None)
    if best_ntree_limit is not None and best_ntree_limit > 0:
        return int(best_ntree_limit)

    return int(model.get_params().get("n_estimators", 1000))


def predict_original_scale(model, X):
    pred = model.predict(X)
    return inverse_transform_pred(pred)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    train_pred = predict_original_scale(model, X_train)
    val_pred = predict_original_scale(model, X_val)

    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)

    print(f"Train MSE: {train_mse:.2f}, RMSE: {np.sqrt(train_mse):.2f}")
    print(f"Val   MSE: {val_mse:.2f}, RMSE: {np.sqrt(val_mse):.2f}")

    return train_pred, val_pred, train_mse, val_mse


# ============================================================
# 5. 全量重训最终模型
# ============================================================
def train_final_model(X: pd.DataFrame, y: pd.Series, best_n_estimators: int):
    """
    用全部训练集重新训练最终模型。
    为避免树数量过少或过多，这里使用验证阶段得到的 best_n_estimators。
    """
    final_n_estimators = max(100, int(best_n_estimators))

    print(f"\n开始使用全部训练集重训 final model，n_estimators={final_n_estimators} ...")

    final_model = make_xgb_model(
        n_estimators=final_n_estimators,
        early_stopping_rounds=None,
    )

    y_fit = transform_target(y)
    final_model.fit(X, y_fit, verbose=False)

    return final_model


# ============================================================
# 6. 结果保存与可视化
# ============================================================
def save_feature_importance(model, feature_cols):
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nXGBoost 特征重要性 Top 20:")
    print(importances.head(20).to_string(index=False))

    importances.to_csv(OUTPUT_IMPORTANCE, index=False)
    print(f"特征重要性已保存至 {OUTPUT_IMPORTANCE}")

    return importances


def save_visualization(y_val, val_pred, train_mse, val_mse, importances):
    print("生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("XGBoost Optimized - Training Visualization", fontsize=16, fontweight="bold")

    # 图1：特征重要性 Top 15
    ax1 = axes[0, 0]
    top15 = importances.head(15)
    ax1.barh(range(len(top15)), top15["importance"].values)
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels(top15["feature"].values)
    ax1.invert_yaxis()
    ax1.set_xlabel("Importance")
    ax1.set_title("Feature Importance (Top 15)")

    # 图2：验证集预测值 vs 真实值
    ax2 = axes[0, 1]
    ax2.scatter(y_val, val_pred, alpha=0.3, s=5)
    ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    ax2.set_xlabel("True cnt")
    ax2.set_ylabel("Predicted cnt")
    ax2.set_title(f"Predicted vs True (Val)\nRMSE={np.sqrt(val_mse):.1f}")

    # 图3：残差分布
    ax3 = axes[1, 0]
    residuals = y_val - val_pred
    ax3.hist(residuals, bins=60, edgecolor="white", alpha=0.8)
    ax3.axvline(0, linestyle="--", lw=2)
    ax3.set_xlabel("Residual (True - Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Residual Distribution\nMean={residuals.mean():.1f}, Std={residuals.std():.1f}")

    # 图4：MSE 对比
    ax4 = axes[1, 1]
    bars = ax4.bar(["Train", "Validation"], [train_mse, val_mse], edgecolor="black")
    ax4.set_ylabel("MSE")
    ax4.set_title(f"Train vs Val MSE\nTrain={train_mse:.1f}, Val={val_mse:.1f}")

    for bar, val in zip(bars, [train_mse, val_mse]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"可视化图片已保存至 {OUTPUT_FIGURE}")


def save_submission(test_ids, test_pred):
    submission = pd.DataFrame({
        "ID": test_ids,
        "cnt": test_pred,
    })

    submission.to_csv(OUTPUT_SUBMISSION, index=False)

    print(f"\n提交文件已保存至 {OUTPUT_SUBMISSION}")
    print(f"提交行数: {len(submission)}")
    print(f"预测 cnt 范围: [{test_pred.min():.1f}, {test_pred.max():.1f}]")
    print("提交 Kaggle 时，如需严格命名，可将 submission_xgboost_optimized.csv 重命名为 submission.csv。")


# ============================================================
# 7. 主程序
# ============================================================
def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练集：{TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"找不到测试集：{TEST_PATH}")

    X, y, X_test, test_ids = prepare_train_test(TRAIN_PATH, TEST_PATH)

    print("清洗与特征提取完成")
    print(f"训练特征矩阵: {X.shape}")
    print(f"测试特征矩阵: {X_test.shape}")
    print(f"特征数: {len(X.columns)}")
    print(f"验证方式: {VALIDATION_MODE}")
    print(f"是否使用 log1p(cnt): {USE_LOG_TARGET}")
    print(f"是否插值修复 windspeed=0: {INTERPOLATE_ZERO_WINDSPEED}")

    X_train, X_val, y_train, y_val = split_train_valid(X, y)
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

    # 1. 先训练验证模型，用于评估和确定最佳树数量
    validation_model = train_validation_model(X_train, y_train, X_val, y_val)

    best_n_estimators = get_best_n_estimators(validation_model)
    print(f"\n验证阶段最佳树数量 best_n_estimators = {best_n_estimators}")

    train_pred, val_pred, train_mse, val_mse = evaluate_model(
        validation_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # 2. 保存验证模型的重要性和可视化，便于报告分析
    importances = save_feature_importance(validation_model, X.columns.tolist())

    save_visualization(
        y_val,
        val_pred,
        train_mse,
        val_mse,
        importances,
    )

    # 3. 用全部训练集重新训练最终模型，再预测测试集
    final_model = train_final_model(X, y, best_n_estimators)

    test_pred = predict_original_scale(final_model, X_test)

    save_submission(test_ids, test_pred)


if __name__ == "__main__":
    main()
