import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

def extract_and_clean_features(df):
    """保持特征工程与训练时绝对一致"""
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

def main():
    print("1. 正在读取全量训练集与测试集...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    # 提取测试集 ID 用于保存提交文件
    test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df['dteday']
    
    # 统一提取特征
    train_df = extract_and_clean_features(train_df)
    test_df = extract_and_clean_features(test_df)
    
    # 准备特征与目标变量
    y_train_full = train_df['cnt']
    cols_to_drop = ['cnt', 'casual', 'registered', 'ID', 'dteday']
    X_train_full = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # 指定类别特征并强转 int，防止预测时报错
    cat_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                    'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak']
    for col in cat_features:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype(int)
            X_test[col] = X_test[col].astype(int)
            
    # 全量标签对数化
    y_train_log = np.log1p(y_train_full)
    
    print("2. 开始使用 100% 全量数据及最优参数训练最终模型 (不再进行早停)...")
    # =========================================================
    # 填入你 Optuna 跑出来的最优参数
    # =========================================================
    final_params = {
        'iterations': 4317,
        'learning_rate': 0.06774373247208128,
        'depth': 4,
        'l2_leaf_reg': 3,
        'subsample': 0.7000932057079796,
        'random_strength': 4.602892427684365,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 200
    }
    
    # 注意：这里直接 fit 全部数据，去掉了 eval_set 和 early_stopping_rounds
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(X_train_full, y_train_log, cat_features=cat_features)
    
    print("\n3. 正在生成测试集预测结果...")
    # 预测出对数结果
    test_pred_log = final_model.predict(X_test)
    
    # 还原真实数值
    test_pred_real = np.expm1(test_pred_log)
    
    # 物理防线：自行车数量不为负，四舍五入为整数
    test_pred_final = np.clip(test_pred_real, a_min=0, a_max=None).round().astype(int)
    
    # 4. 保存为提交格式
    output_filename = "final_optuna_submission.csv"
    submission = pd.DataFrame({
        'ID' if isinstance(test_ids.iloc[0], (int, np.integer)) else 'dteday': test_ids,
        'cnt': test_pred_final
    })
    submission.to_csv(output_filename, index=False)
    
    print("\n" + "=" * 50)
    print(f"🎉 大功告成！测试集预测文件已保存为：{output_filename}")
    print("祝你在网站上取得极其惊艳的排名！冲鸭！")
    print("=" * 50)

if __name__ == "__main__":
    main()