import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. 升级版特征工程函数
# =========================================================
def extract_and_clean_features(df):
    df = df.copy()
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['weathersit'] = df['weathersit'].replace(4, 3)
    
    # 气象交叉特征
    df["temp_diff"] = df["temp"] - df["atemp"]
    df["temp_hum"] = df["temp"] * df["hum"] # 🌟 新增：温湿度交叉特征
    
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
    
    # 🌟 新增：工作日与小时的强交叉特征 (转为字符串类别)
    df["workingday_hr"] = df["workingday"].astype(str) + "_" + df["hr"].astype(str)
    
    return df

# =========================================================
# 2. 🌟 新增：样本权重生成器
# =========================================================
def generate_sample_weights(df):
    weights = np.ones(len(df))
    # 惩罚恶劣天气预测不准：将 weathersit >= 3 的样本权重调高
    weights[df['weathersit'] >= 3] += 1.5 
    # 惩罚高峰期预测不足：直接定位 8点, 17点, 18点的工作日
    peak_mask = (df['workingday'] == 1) & (df['hr'].isin([8, 17, 18]))
    weights[peak_mask] += 1.5
    return weights

# =========================================================
# 3. 准备全局数据
# =========================================================
print("正在分别读取并清洗训练集(train.csv)和验证集(hour.csv)...")

train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("hour.csv")

train_df = extract_and_clean_features(train_df)
valid_df = extract_and_clean_features(valid_df)

y_train = train_df['cnt']
X_train = train_df.drop(columns=['cnt', 'casual', 'registered', 'ID', 'dteday'], errors='ignore')

y_valid = valid_df['cnt']
X_valid = valid_df.drop(columns=['cnt', 'casual', 'registered', 'ID', 'dteday'], errors='ignore')

# 提取训练集的样本权重
train_weights = generate_sample_weights(X_train)

# 更新类别特征列表
cat_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'workingday_hr']

# 处理类别类型 (跳过字符串类型的 workingday_hr)
for col in cat_features:
    if col != 'workingday_hr':
        if col in X_train.columns: X_train[col] = X_train[col].astype(int)
        if col in X_valid.columns: X_valid[col] = X_valid[col].astype(int)

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

# =========================================================
# 4. Optuna 自动优化的目标函数
# =========================================================
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 5000), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 5, 8), # 略微增加深度以拟合复杂特征
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
    
    # 🌟 传入样本权重 sample_weight
    model.fit(
        X_train, y_train_log, 
        sample_weight=train_weights,
        cat_features=cat_features, 
        eval_set=(X_valid, y_valid_log),
        use_best_model=True
    )
    
    y_pred_log = model.predict(X_valid)
    y_pred_real = np.expm1(y_pred_log)
    y_pred_final = np.clip(y_pred_real, a_min=0, a_max=None).round().astype(int)
    
    mse = mean_squared_error(y_valid, y_pred_final)
    return mse

if __name__ == "__main__":
    print("\n🚀 启动全自动参数搜索 (Optuna)...")
    study = optuna.create_study(direction='minimize', study_name='CatBoost_Bike')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "=" * 50)
    print(f"🥇 验证集取得的最低 MSE: {study.best_value:.2f}")
    print("\n💡 你应该直接抄作业的【完美参数组合】是：")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("=" * 50)