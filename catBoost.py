import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

def extract_and_clean_features(df):
    df = df.copy()
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['weathersit'] = df['weathersit'].replace(4, 3)
    
    df["temp_diff"] = df["temp"] - df["atemp"]
    df["temp_hum"] = df["temp"] * df["hum"] # 🌟 保持一致
    
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
    
    # 🌟 保持一致：工作日与小时的强交叉特征
    df["workingday_hr"] = df["workingday"].astype(str) + "_" + df["hr"].astype(str)
    
    return df

# 🌟 新增权重生成器
def generate_sample_weights(df):
    weights = np.ones(len(df))
    weights[df['weathersit'] >= 3] += 1.5 
    peak_mask = (df['workingday'] == 1) & (df['hr'].isin([8, 17, 18]))
    weights[peak_mask] += 1.5
    return weights

def main():
    print("1. 正在读取全量训练集与测试集...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df['dteday']
    
    train_df = extract_and_clean_features(train_df)
    test_df = extract_and_clean_features(test_df)
    
    y_train_full = train_df['cnt']
    cols_to_drop = ['cnt', 'ID', 'dteday']
    X_train_full = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # 全量数据样本权重
    train_weights = generate_sample_weights(X_train_full)
    
    cat_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                    'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'workingday_hr']
    for col in cat_features:
        if col != 'workingday_hr':
            if col in X_train_full.columns: X_train_full[col] = X_train_full[col].astype(int)
            if col in X_test.columns: X_test[col] = X_test[col].astype(int)
            
    y_train_log = np.log1p(y_train_full)
    
    print("2. 开始使用 100% 全量数据及最优参数训练最终模型...")
    # =========================================================
    # 🚨 注意：这里记得填入你用新的 auto.py 跑出来的最佳参数！
    # =========================================================
    final_params = {
        'iterations': 4156,
        'learning_rate': 0.08106663807261587,
        'depth': 5,
        'l2_leaf_reg': 15,
        'subsample': 0.8189100188766656,
        'random_strength': 2.536451384863532,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 200
    }
    
    final_model = CatBoostRegressor(**final_params)
    # 🌟 在全量训练时同样应用样本权重
    final_model.fit(X_train_full, y_train_log, cat_features=cat_features, sample_weight=train_weights)
    
    print("\n3. 正在生成测试集预测结果...")
    test_pred_log = final_model.predict(X_test)
    test_pred_real = np.expm1(test_pred_log)
    test_pred_final = np.clip(test_pred_real, a_min=0, a_max=None).round().astype(int)
    
    output_filename = "final_optuna_submission.csv"
    submission = pd.DataFrame({
        'ID' if isinstance(test_ids.iloc[0], (int, np.integer)) else 'dteday': test_ids,
        'cnt': test_pred_final
    })
    submission.to_csv(output_filename, index=False)
    
    print("\n" + "=" * 50)
    print(f"🎉 大功告成！预测文件已保存为：{output_filename}")
    print("=" * 50)

if __name__ == "__main__":
    main()