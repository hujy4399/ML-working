import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（如果是 Windows，可使用 SimHei；Mac 可使用 Arial Unicode MS 或 Heiti TC）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def train_lightgbm_log_model(file_path='train_washed.csv'):
    print("1. 正在加载清洗后的数据...")
    df = pd.read_csv(file_path, index_col='dteday', parse_dates=True)
    
    # 2. 剥离特征 (X) 和目标 (y)
    # 务必把 cnt, casual, registered 去掉，防止数据泄露
    X = df.drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
    y = df['cnt']
    
    # 3. 划分训练集和测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # =========================================================
    # 【新增：取对数】对目标变量 y 进行 log(1+x) 转换
    # 解决右偏长尾分布问题，让模型学习更稳定
    # =========================================================
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # 4. 初始化 LightGBM 回归器
    model = lgb.LGBMRegressor(
        n_estimators=1000,       
        learning_rate=0.05,      
        max_depth=8,             
        num_leaves=31,           
        subsample=0.8,           
        colsample_bytree=0.8,    
        random_state=42,
        n_jobs=-1                
    )
    
    print("\n2. 开始训练 LightGBM 模型 (使用对数化后的标签)...")
    # 早停法监控：注意这里喂给验证集的也是取过对数的 y_test_log
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)], 
        eval_metric='rmse',          
        callbacks=callbacks
    )
    
    # 5. 模型预测与评估
    print("\n3. 模型评估中...")
    
    # 模型输出的是“对数级别的预测值”
    y_pred_log = model.predict(X_test)
    
    # =========================================================
    # 【新增：对数还原】将预测结果用 exp(x)-1 还原为真实的租借数量
    # =========================================================
    y_pred_real = np.expm1(y_pred_log)
    
    # 计算误差时，使用原始真实的 y_test 和还原后的预测值进行对比
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_real))
    r2 = r2_score(y_test, y_pred_real)
    
    print(f"========== 最终评估结果 ==========")
    print(f"测试集 RMSE (均方根误差): {rmse:.2f} 辆")
    print(f"测试集 R² (决定系数):    {r2:.4f}")
    print(f"==================================")
    
    # 6. 特征重要性可视化
    print("\n4. 绘制特征重要性图表...")
    plot_feature_importance(model, X_train.columns)
    
    return model

def plot_feature_importance(model, feature_names):
    """提取并绘制 LightGBM 的特征重要性"""
    importance = model.booster_.feature_importance(importance_type='gain')
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    top_n = 20
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(top_n), palette='viridis')
    plt.title(f'LightGBM 特征重要性 Top {top_n} (按 Gain 计算)')
    plt.xlabel('信息增益 (Gain)')
    plt.ylabel('特征名称')
    plt.tight_layout()
    plt.show()

def predict_and_save(model, test_file_path='test.csv', output_file='submission.csv'):
    """
    读取未清洗的 test.csv，进行与训练集一致的清洗，并输出预测结果。
    """
    print(f"\n--- 开始处理测试集: {test_file_path} ---")
    df_test = pd.read_csv(test_file_path)
    
    # 1. 提取标识符用于最终的输出文件
    # 如果有 ID 就用 ID，没有就用时间 dteday
    if 'ID' in df_test.columns:
        result_df = pd.DataFrame({'ID': df_test['ID']})
        df_test = df_test.drop(columns=['ID'])
    else:
        result_df = pd.DataFrame({'dteday': df_test['dteday']})
        
    # 2. 复刻训练集的数据清洗逻辑
    df_test['dteday'] = pd.to_datetime(df_test['dteday'])
    df_test['day_of_month'] = df_test['dteday'].dt.day
    df_test.set_index('dteday', inplace=True)
    
    df_test['windspeed'] = df_test['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df_test['hum'] = df_test['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df_test['weathersit'] = df_test['weathersit'].replace(4, 3)
    
    # 独热编码
    categorical_cols = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
    
    for col in df_test.columns:
        if df_test[col].dtype == 'bool':
            df_test[col] = df_test[col].astype(int)
            
    # 剔除可能存在的泄漏列（有些测试集会把 cnt 填为 0 或空值）
    df_test = df_test.drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
    
    # =========================================================
    # 【核心操作：特征对齐】
    # 获取模型训练时记住的所有列名 (model.feature_name_)
    # 使用 reindex 强行把测试集的列变得和训练集一模一样。
    # 如果测试集少了某列，自动用 0 补齐；多出的垃圾列，自动删掉。
    # =========================================================
    train_features = model.feature_name_
    df_test = df_test.reindex(columns=train_features, fill_value=0)
    
    print("清洗与特征对齐完成！正在进行预测...")
    
    # 3. 模型预测 (模型吐出的是对数值 log)
    y_pred_log = model.predict(df_test)
    
    # 4. 对数还原 (用 expm1 还原为真实的租车数量)
    y_pred_real = np.expm1(y_pred_log)
    
    # 5. 后处理：现实中自行车数量不可能是负数，且必须是整数
    # clip(lower=0) 保证最小值为 0；round() 四舍五入取整
    y_pred_final = np.clip(y_pred_real, a_min=0, a_max=None).round().astype(int)
    
    # 保存至结果 DataFrame
    result_df['cnt'] = y_pred_final
    
    # 6. 导出到 CSV 文件
    result_df.to_csv(output_file, index=False)
    print(f"预测完毕！结果已成功保存至: {output_file}")
    
    return result_df

if __name__ == "__main__":
    # 1. 训练模型 (假设你之前定义了 train_lightgbm_log_model 函数)
    print(">>> 第一阶段：训练模型 <<<")
    # 如果 train_washed.csv 不存在，这里要确保你先运行了数据清洗脚本
    my_model = train_lightgbm_log_model('train_washed.csv')
    
    # 2. 对未处理的 test.csv 进行预测并生成结果
    print("\n>>> 第二阶段：生成测试集预测 <<<")
    # 假设你的未处理测试集文件名为 test.csv
    predict_and_save(my_model, test_file_path='test.csv', output_file='submission.csv')