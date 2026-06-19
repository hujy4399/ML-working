import pandas as pd
import numpy as np

# 季节独热编码 ： 000 春 100 夏 010 秋 001 冬
# mnth（月份），weathersit（天气），hr（小时）和 weekday（星期）全零为第一个，否则为置1位
def clean_and_save_data(input_path, output_path):
    """
    读取原始数据进行清洗，并由你控制输出到指定的 CSV 文件。
    """
    print(f"开始读取原始数据: {input_path} ...")
    df = pd.read_csv(input_path)
    
    # 1. 移除无预测价值的列
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # 2. 日期时间处理
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day_of_month'] = df['dteday'].dt.day
    df.set_index('dteday', inplace=True)
    
    # 3. 异常值/隐性缺失值处理 (利用前后值进行线性插值修复 0.0)
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    
    # 4. 处理极端天气 (4并入3)
    df['weathersit'] = df['weathersit'].replace(4, 3)
    
    # 5. 类别特征转换 (独热编码 One-Hot)
    categorical_cols = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 6. 数据类型整理 (将新版本的 bool 类型统一转为 int)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            
    # 7. 控制输出：将清洗后的 DataFrame 保存到指定的本地路径
    print(f"清洗完成！正在保存至输出路径: {output_path} ...")
    df.to_csv(output_path)
    print("文件保存成功！\n")
    
    return output_path

def load_washed_data(file_path):
    """
    【队友调用的接口】
    读取清洗后的数据，直接返回特征集 X 和目标标签 y，供机器学习模型使用。
    """
    df = pd.read_csv(file_path, index_col='dteday', parse_dates=True)
    
    # 拆分特征(X)和预测目标(y)
    # 此处假设你的目标是预测总数(cnt)，所以顺便剥离了 casual 和 registered 两个泄漏变量
    X = df.drop(columns=['cnt', 'casual', 'registered'], errors='ignore')
    y = df['cnt']
    
    return X, y

# ==========================================
# 你的主控程序
# ==========================================
if __name__ == "__main__":
    
    # ---> 在这里控制你的输入和输出 <---
    INPUT_FILE = 'train.csv'                # 原始数据的路径
    OUTPUT_FILE = 'train_washed.csv'        # 你希望输出的清洗后文件路径
    
    # 1. 由你来执行清洗并生成文件
    clean_and_save_data(input_path=INPUT_FILE, output_path=OUTPUT_FILE)
    
    # 2. 你的队友拿走 load_washed_data 接口后，他们那边的调用演示：
    print("--- 模拟队友调用接口 ---")
    X, y = load_washed_data(file_path=OUTPUT_FILE)
    
    print(f"特征集 X 的形状: {X.shape}")
    print(f"目标集 y 的形状: {y.shape}")
    print("数据加载完毕，队友可以开始用 X 和 y 训练模型了！")