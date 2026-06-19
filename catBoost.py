import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def extract_and_clean_features(df):
    """
    融合了 select(1).py 和 图片建议 的综合特征工程模块
    """
    df = df.copy()
    
    # === 1. 基础异常值清洗 ===
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['weathersit'] = df['weathersit'].replace(4, 3)
    
    # === 2. 汲取图片建议：体感温度差 ===
    df["temp_diff"] = df["temp"] - df["atemp"]
    
    # === 3. 融合 select(1).py：高级时间特征 ===
    date = pd.to_datetime(df["dteday"])
    
    df["day_of_month"] = date.dt.day
    df["day_of_year"] = date.dt.dayofyear
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)
    
    # 距离运营首日的天数 (捕捉长期增长趋势)
    df["days_since_start"] = (date - pd.Timestamp("2011-01-01")).dt.days
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # 小时、星期和月份都是周期变量，用正余弦表示首尾相邻关系
    for column, period in (("hr", 24), ("weekday", 7), ("mnth", 12)):
        angle = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)

    # 工作日上下班高峰识别
    df["is_morning_peak"] = (df["workingday"].eq(1) & df["hr"].between(7, 9)).astype(int)
    df["is_evening_peak"] = (df["workingday"].eq(1) & df["hr"].between(17, 19)).astype(int)
    
    return df

def main():
    print("1. 正在读取并进行进阶特征工程...")
    train_raw = pd.read_csv("train.csv")
    test_raw = pd.read_csv("test.csv")
    
    # 提取测试集 ID 用于最终提交
    test_ids = test_raw['ID'] if 'ID' in test_raw.columns else test_raw['dteday']
    
    # 应用特征工程
    df_train = extract_and_clean_features(train_raw)
    df_test = extract_and_clean_features(test_raw)
    
    # 剥离无用列和目标变量
    cols_to_drop = ["ID", "dteday", "cnt", "casual", "registered"]
    X = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    y = df_train["cnt"]
    X_test_final = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # 声明所有类别特征 (包括新提取的 0/1 特征)
    cat_features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
        'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak'
    ]
    
    # 确保类别特征是 int 类型
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(int)
            X_test_final[col] = X_test_final[col].astype(int)

    # 划分验证集用于早停
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 对数化目标变量
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)
    
    print("\n2. 特征已就绪，开始训练 CatBoost 模型...")
    # 沿用我们之前配置好的防过拟合参数
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        subsample=0.8,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train_log, 
        cat_features=cat_features, 
        eval_set=(X_valid, y_valid_log),
        use_best_model=True
    )
    
    print("\n3. 正在预测测试集并保存结果...")
    y_pred_log = model.predict(X_test_final)
    y_pred_real = np.expm1(y_pred_log)
    
    # 兜底处理
    y_pred_final = np.clip(y_pred_real, a_min=0, a_max=None).round().astype(int)
    
    # 导出文件
    output_file = 'advanced_submission.csv'
    result_df = pd.DataFrame({
        'ID' if isinstance(test_ids.iloc[0], (int, np.integer)) else 'dteday': test_ids,
        'cnt': y_pred_final
    })
    result_df.to_csv(output_file, index=False)
    
    print(f"\n🎉 完美融合！进阶预测结果已保存至: {output_file}")

if __name__ == "__main__":
    main()