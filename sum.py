import pandas as pd
import numpy as np

def ultimate_three_model_blend():
    print("1. 正在读取三个模型的预测结果...")
    
    # 读取三个提交文件
    xgb_sub = pd.read_csv('submission_xgboost_high_demand_calibrated.csv')
    cb_sub = pd.read_csv('final_optuna_submission.csv')
    nn_sub = pd.read_csv('submission_nn_plus.csv')
    
    # 以测试集的 ID 为基准合并
    df = xgb_sub.rename(columns={'cnt': 'cnt_xgb'})
    df = df.merge(cb_sub.rename(columns={'cnt': 'cnt_cb'}), on='ID', how='inner')
    df = df.merge(nn_sub.rename(columns={'cnt': 'cnt_nn'}), on='ID', how='inner')
    
    # 建立探针：取三者的平均值来判断当前属于什么“段位”
    df['mean_pred'] = (df['cnt_xgb'] + df['cnt_cb'] + df['cnt_nn']) / 3.0

    print("2. 正在执行三模型动态条件融合...")
    
    def apply_ultimate_weights(row):
        avg = row['mean_pred']
        
        # 提取各个模型的当前预测值
        val_cb = row['cnt_cb']
        val_xgb = row['cnt_xgb']
        val_nn = row['cnt_nn']
        
        if avg < 249:
            # 策略 A：冷清时段 (CatBoost 主导，NN 辅助)
            return (val_cb * 0.591) + (val_xgb * 0.140) + (val_nn * 0.269)
            
        elif avg <= 532:
            # 策略 B：常规时段 (平稳过渡，平均中和噪音)
            return (val_cb * 0.411) + (val_xgb * 0.281) + (val_nn * 0.308)
            
        else:
            # 策略 C：大爆发高峰期 (XGBoost 主导)
            return (val_cb * 0.297) + (val_xgb * 0.608) + (val_nn * 0.095)

    # 涂上魔法！
    df['cnt'] = df.apply(apply_ultimate_weights, axis=1)
    
    # 物理防线：单车数量不能为负数，且必须是整数
    df['cnt'] = np.clip(df['cnt'], a_min=0, a_max=None).round().astype(int)
    
    print("\n3. 保存结果...")
    final_submission = df[['ID', 'cnt']]
    out_name = "ultimate_three_model_blend.csv"
    final_submission.to_csv(out_name, index=False)
    
    print("=" * 50)
    print(f"🎉 终极融合完成！(预期线下 MSE: 2614左右)")
    print(f"文件已保存为: {out_name}")
    print("=" * 50)

if __name__ == "__main__":
    ultimate_three_model_blend()