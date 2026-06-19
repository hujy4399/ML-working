import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. 准备工作：把队友跑出来的数据填到这里
# ==========================================
print("1. 正在读取四个模型的【本地验证集预测】和【测试集预测】...")

# 【注意】这里需要你用真实代码去读取你们本地切分出来的最后 20% 的真实 cnt！
# 假设你已经保存了一个 val_true.csv
# y_val_true = pd.read_csv("val_true.csv")['cnt'].values
y_val_true = np.random.rand(2000) * 1000 # [仅做演示占位，请替换为真实标签]

# --- 读取四个模型在【验证集】上的预测结果 ---
# (让队友把 evaluate_model 里的 val_pred 保存给你)
# val_xgb = pd.read_csv("val_xgb.csv")['cnt'].values
# val_cat = pd.read_csv("val_cat.csv")['cnt'].values
# val_rf = pd.read_csv("val_rf.csv")['cnt'].values
# val_ridge = pd.read_csv("val_ridge.csv")['cnt'].values

# [演示占位数据]
val_xgb = np.random.rand(2000) * 1000  
val_cat = np.random.rand(2000) * 1000  
val_rf = np.random.rand(2000) * 1000   
val_ridge = np.random.rand(2000) * 1000 

# ==========================================
# 2. 核心科技：暴力搜索最佳融合权重
# ==========================================
print("2. 正在多维空间中暴力搜索黄金融合比例...")
best_mse = float('inf')
best_weights = (0, 0, 0, 0)

# 步长 0.05，遍历所有可能的权重组合 (确保总和为 1)
for w_xgb in np.arange(0, 1.05, 0.05):
    for w_cat in np.arange(0, 1.05 - w_xgb, 0.05):
        for w_rf in np.arange(0, 1.05 - w_xgb - w_cat, 0.05):
            w_ridge = 1.0 - w_xgb - w_cat - w_rf
            
            # 剔除由于浮点精度产生的微小负数
            if w_ridge < -1e-5: 
                continue
            w_ridge = max(0, w_ridge)
            
            # 加权融合验证集的预测
            blend_val = (w_xgb * val_xgb + 
                         w_cat * val_cat + 
                         w_rf * val_rf + 
                         w_ridge * val_ridge)
            
            mse = mean_squared_error(y_val_true, blend_val)
            
            # 如果出现更低的 MSE，就更新记录
            if mse < best_mse:
                best_mse = mse
                best_weights = (w_xgb, w_cat, w_rf, w_ridge)

print(f"\n🎉 搜索完毕！本地极限验证集 MSE: {best_mse:.4f}")
print("🏆 最优权重分配如下：")
print(f"   XGBoost (主攻手):  {best_weights[0]:.2f}")
print(f"   CatBoost (主攻手): {best_weights[1]:.2f}")
print(f"   RandomForest(防守盾): {best_weights[2]:.2f}")
print(f"   RidgeCV  (拉升器): {best_weights[3]:.2f}")

# ==========================================
# 3. 终极合成：生成向 Kaggle 提交的最终文件
# ==========================================
print("\n3. 正在应用最佳权重，生成 FINAL_SUBMISSION.csv...")

# --- 读取四个模型在【测试集】上的预测结果 ---
# sub_xgb = pd.read_csv("submission_xgboost_optimized.csv")['cnt'].values
# sub_cat = pd.read_csv("submission_catboost.csv")['cnt'].values
# sub_rf = pd.read_csv("submission_rf.csv")['cnt'].values
# sub_ridge = pd.read_csv("submission_ridge.csv")['cnt'].values
# test_ids = pd.read_csv("submission_xgboost_optimized.csv")['ID'].values

# [演示占位数据]
sub_xgb = np.random.rand(3476) * 1000  
sub_cat = np.random.rand(3476) * 1000  
sub_rf = np.random.rand(3476) * 1000   
sub_ridge = np.random.rand(3476) * 1000 
test_ids = np.arange(1, 3477)

# 使用刚刚找到的最优权重进行加权相加
final_test_pred = (best_weights[0] * sub_xgb + 
                   best_weights[1] * sub_cat + 
                   best_weights[2] * sub_rf + 
                   best_weights[3] * sub_ridge)

# 强制确保没有负数租借量
final_test_pred = np.clip(final_test_pred, a_min=0, a_max=None)

# 保存文件
final_submission = pd.DataFrame({
    'ID': test_ids,
    'cnt': final_test_pred
})

# final_submission.to_csv("FINAL_SUBMISSION.csv", index=False)
print("🚀 大功告成！FINAL_SUBMISSION.csv 已准备完毕，去提交吧！")