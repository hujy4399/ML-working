import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. 之前的高级特征工程函数 (保持不变)
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
X = df.drop(columns=['cnt', 'casual', 'registered', 'ID', 'dteday'], errors='ignore')

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
    """
    Optuna 会在设定的范围内，自动给你推荐一组参数 (trial)
    """
    params = {
        # 1. 给足上限，让它跑个痛快
        'iterations': trial.suggest_int('iterations', 2000, 5000), 
        
        # 2. 学习率适当放宽，允许大步试错
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        
        # 3. 压低深度！让他不要死记硬背
        'depth': trial.suggest_int('depth', 4, 7),
        
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 15),
        
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        
        # 🌟 4. 加入终极防噪音魔法！搜索范围从 1 到 10
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
        
        'loss_function': 'RMSE', # 或者换回 Poisson 杀伤力更大
        'eval_metric': 'RMSE',
        
        # 5. 放宽早停耐心值，允许震荡
        'early_stopping_rounds': 150, 
        
        'random_seed': 42,
        'verbose': False
    }
    
    # 用这组随机生成的参数建立模型
    model = CatBoostRegressor(**params)
    
    # 训练模型
    model.fit(
        X_train, y_train_log, 
        cat_features=cat_features, 
        eval_set=(X_valid, y_valid_log),
        use_best_model=True
    )
    
    # 预测并还原
    y_pred_log = model.predict(X_valid)
    y_pred_real = np.expm1(y_pred_log)
    y_pred_final = np.clip(y_pred_real, a_min=0, a_max=None).round().astype(int)
    
    # 计算真实 MSE
    mse = mean_squared_error(y_valid, y_pred_final)
    
    # 把 MSE 交给 Optuna，让它记住这组参数的好坏
    return mse

# =========================================================
# 4. 启动自动化搜索任务
# =========================================================
if __name__ == "__main__":
    print("\n🚀 启动全自动参数搜索 (Optuna)...")
    print("你可以去喝杯咖啡，机器大概会测试 30 种不同的参数组合...")
    
    # direction='minimize' 表示我们要让返回的 MSE 越小越好
    study = optuna.create_study(direction='minimize', study_name='CatBoost_Bike')
    
    # 开始搜索！n_trials=30 表示测试 30 组不同的参数 (电脑快的话可以改成 50)
    study.optimize(objective, n_trials=30)
    
    # 搜索结束，输出成绩单！
    print("\n" + "=" * 50)
    print("🏆 自动化调参结束！")
    print("=" * 50)
    print(f"🥇 验证集取得的最低 MSE: {study.best_value:.2f}")
    print(f"🥇 验证集取得的最低 RMSE: {np.sqrt(study.best_value):.2f} 辆")
    print("\n💡 你应该直接抄作业的【完美参数组合】是：")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("=" * 50)