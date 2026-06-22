import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. 高级特征工程函数 (保持不变)
# =========================================================
def extract_and_clean_features(df):
    df = df.copy()
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['weathersit'] = df['weathersit'].replace(4, 3)
    df["temp_diff"] = df["temp"] - df["atemp"]
    
    df['dteday'] = pd.to_datetime(df['dteday'])
    df = df.sort_values(by=['dteday', 'hr']).reset_index(drop=True)
    
    df["day_of_month"] = df['dteday'].dt.day
    df["day_of_year"] = df['dteday'].dt.dayofyear
    df["week_of_year"] = df['dteday'].dt.isocalendar().week.astype(int)
    df["days_since_start"] = (df['dteday'] - pd.Timestamp("2011-01-01")).dt.days
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    for column, period in (("hr", 24), ("weekday", 7), ("mnth", 12)):
        angle = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)

    df["is_morning_peak"] = (df["workingday"].eq(1) & df["hr"].between(7, 9)).astype(int)
    df["is_evening_peak"] = (df["workingday"].eq(1) & df["hr"].between(17, 19)).astype(int)
    return df

# =========================================================
# 2. 准备全局数据
# =========================================================
print("正在读取并清洗数据...")
df = pd.read_csv("train.csv")
df = extract_and_clean_features(df)

y = df['cnt']
# 原本的代码中有 casual 和 registered，如果你的数据里没有，ignore 也不会报错，这里为了干净直接去掉它们
X = df.drop(columns=['cnt', 'ID', 'dteday'], errors='ignore')

cat_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak']
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype(int)

# 严格按时间顺序切分 (80% 训练，20% 验证)
split_idx = int(len(X) * 0.8)
X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

# =========================================================
# 3. 🎯 核心：定义 Optuna 自动优化的目标函数
# =========================================================
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 5000), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 7),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
        'loss_function': 'RMSE', 
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 150, 
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    
    model.fit(
        X_train, y_train_log, 
        cat_features=cat_features, 
        eval_set=(X_valid, y_valid_log),
        use_best_model=True
    )
    
    # 🌟【修复点】：获取早停触发时，模型真正觉得最好的迭代轮数，并告诉 Optuna
    actual_best_iteration = model.get_best_iteration()
    trial.set_user_attr("actual_best_iter", actual_best_iteration)
    
    # 预测并还原
    y_pred_log = model.predict(X_valid)
    y_pred_real = np.expm1(y_pred_log)
    y_pred_final = np.clip(y_pred_real, a_min=0, a_max=None).round().astype(int)
    
    # 计算真实 MSE
    mse = mean_squared_error(y_valid, y_pred_final)
    return mse

# =========================================================
# 4. 启动自动化搜索任务
# =========================================================
if __name__ == "__main__":
    print("\n🚀 启动全自动参数搜索 (Optuna)...")
    print("你可以去喝杯咖啡，机器大概会测试 30 种不同的参数组合...")
    
    study = optuna.create_study(direction='minimize', study_name='CatBoost_Bike')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "=" * 50)
    print("🏆 自动化调参结束！")
    print("=" * 50)
    
    best_trial = study.best_trial
    # 🌟 提取我们刚刚保存的真实迭代次数
    real_iterations = best_trial.user_attrs["actual_best_iter"]
    
    print(f"🥇 验证集取得的最低 MSE: {study.best_value:.2f}")
    print(f"🥇 验证集取得的最低 RMSE: {np.sqrt(study.best_value):.2f} 辆")
    print("\n💡 你应该直接抄作业的【完美参数组合】是：")
    
    # 过滤掉原本虚高的 iterations，换成真实的
    for key, value in best_trial.params.items():
        if key != 'iterations':
            print(f"    '{key}': {value},")
            
    print(f"    'iterations': {real_iterations},  <-- 🌟 这才是真正触发早停的最佳轮数！")
    print("=" * 50)