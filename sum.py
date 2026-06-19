import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================================================
# 1. 统一的高级特征工程 (吸取三个代码的精华)
# =========================================================
def extract_features(df):
    """提取时间、天气等高级特征"""
    df = df.copy()
    
    # 基础清洗
    df['windspeed'] = df['windspeed'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['hum'] = df['hum'].replace(0.0, np.nan).interpolate(method='linear').bfill().ffill()
    df['weathersit'] = df['weathersit'].replace(4, 3)
    df["temp_diff"] = df["temp"] - df["atemp"]
    
    # 时间特征
    date = pd.to_datetime(df["dteday"])
    df["day_of_month"] = date.dt.day
    df["day_of_year"] = date.dt.dayofyear
    df["week_of_year"] = date.dt.isocalendar().week.astype(int)
    df["days_since_start"] = (date - pd.Timestamp("2011-01-01")).dt.days
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # 周期正余弦编码
    for column, period in (("hr", 24), ("weekday", 7), ("mnth", 12)):
        angle = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)

    # 早晚高峰
    df["is_morning_peak"] = (df["workingday"].eq(1) & df["hr"].between(7, 9)).astype(int)
    df["is_evening_peak"] = (df["workingday"].eq(1) & df["hr"].between(17, 19)).astype(int)
    
    return df

def main():
    print(">>> 阶段 1: 数据合并与特征工程")
    train_raw = pd.read_csv("local_train.csv")
    test_raw = pd.read_csv("local_test.csv")
    
    # 保存测试集ID
    test_ids = test_raw['ID'] if 'ID' in test_raw.columns else test_raw['dteday']
    y_train_full = train_raw['cnt']
    
    # 为了保证独热编码维度一致，将 train 和 test 拼接到一起处理
    train_raw['is_test'] = 0
    test_raw['is_test'] = 1
    df_all = pd.concat([train_raw, test_raw], ignore_index=True)
    
    # 提取特征
    df_all = extract_features(df_all)
    cols_to_drop = ["ID", "dteday", "cnt", "casual", "registered"]
    df_all = df_all.drop(columns=[c for c in cols_to_drop if c in df_all.columns])
    
    # 类别特征列表
    cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'is_weekend', 'is_morning_peak', 'is_evening_peak']

    # =========================================================
    # 2. 双轨制数据分配 (为不同模型定制口味)
    # =========================================================
    print(">>> 阶段 2: 构建模型专属特征矩阵")
    # 【版本 A】：给 CatBoost 和 LightGBM (保留原始类别)
    X_tree = df_all.copy()
    for col in cat_cols:
        X_tree[col] = X_tree[col].astype(int)
    
    # 【版本 B】：给 XGBoost (进行独热编码 One-Hot)
    X_xgb = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)
    
    # 拆分回 train 和 test
    train_idx = df_all[df_all['is_test'] == 0].index
    test_idx = df_all[df_all['is_test'] == 1].index
    
    X_tree_train, X_tree_test = X_tree.loc[train_idx].drop('is_test', axis=1), X_tree.loc[test_idx].drop('is_test', axis=1)
    X_xgb_train, X_xgb_test = X_xgb.loc[train_idx].drop('is_test', axis=1), X_xgb.loc[test_idx].drop('is_test', axis=1)
    
    # 划分验证集 (为了保证三个模型在同一批数据上验证，我们划分 index)
    train_sub_idx, valid_sub_idx = train_test_split(range(len(train_idx)), test_size=0.2, random_state=42)
    
    y_log = np.log1p(y_train_full)
    y_tr_log, y_va_log = y_log.iloc[train_sub_idx], y_log.iloc[valid_sub_idx]
    
    # =========================================================
    # 3. 训练三剑客 (CatBoost / LightGBM / XGBoost)
    # =========================================================
    print("\n>>> 阶段 3: 训练 [CatBoost] 模型...")
    cat_model = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5, subsample=0.8,
        loss_function='RMSE', random_seed=42, verbose=False
    )
    cat_model.fit(
        X_tree_train.iloc[train_sub_idx], y_tr_log,
        cat_features=cat_cols,
        eval_set=(X_tree_train.iloc[valid_sub_idx], y_va_log),
        early_stopping_rounds=50
    )
    
    print(">>> 阶段 3: 训练 [LightGBM] 模型...")
    # 转换类型为 category，LGBM最喜欢这种格式
    X_lgb_train = X_tree_train.copy()
    for col in cat_cols: X_lgb_train[col] = X_lgb_train[col].astype('category')
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8,
        random_state=42, n_jobs=-1
    )
    lgb_model.fit(
        X_lgb_train.iloc[train_sub_idx], y_tr_log,
        categorical_feature=cat_cols,
        eval_set=[(X_lgb_train.iloc[valid_sub_idx], y_va_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    print(">>> 阶段 3: 训练 [XGBoost] 模型...")
    xgb_model = XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=1, random_state=42, n_jobs=-1
    )
    xgb_model.fit(
        X_xgb_train.iloc[train_sub_idx], y_tr_log,
        eval_set=[(X_xgb_train.iloc[valid_sub_idx], y_va_log)],
        verbose=False
    )

    # =========================================================
    # 4. 预测与验证集表现对比
    # =========================================================
    print("\n>>> 阶段 4: 模型表现对比")
    def evaluate(model, X_val, name):
        pred_log = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_va_log), np.expm1(pred_log)))
        print(f"[{name}] 验证集真实 RMSE: {rmse:.2f}")
        return pred_log

    pred_va_cat = evaluate(cat_model, X_tree_train.iloc[valid_sub_idx], "CatBoost")
    pred_va_lgb = evaluate(lgb_model, X_lgb_train.iloc[valid_sub_idx], "LightGBM")
    pred_va_xgb = evaluate(xgb_model, X_xgb_train.iloc[valid_sub_idx], "XGBoost ")
    
    # =========================================================
    # 5. 最终预测与加权融合 (Blending)
    # =========================================================
    print("\n>>> 阶段 5: 测试集预测与加权融合")
    
    # 获取测试集的独立预测值 (并还原对数)
    X_lgb_test = X_tree_test.copy()
    for col in cat_cols: X_lgb_test[col] = X_lgb_test[col].astype('category')
        
    test_pred_cat = np.expm1(cat_model.predict(X_tree_test))
    test_pred_lgb = np.expm1(lgb_model.predict(X_lgb_test))
    test_pred_xgb = np.expm1(xgb_model.predict(X_xgb_test))
    
    # 💡 在这里调整三个模型的权重！权重加起来要等于 1.0
    # 通常把验证集 RMSE 最低的模型权重设大一点
    weight_cat = 0.40
    weight_lgb = 0.20
    weight_xgb = 0.40
    
    print(f"融合权重设定 -> CatBoost:{weight_cat}, LightGBM:{weight_lgb}, XGBoost:{weight_xgb}")
    
    final_pred = (test_pred_cat * weight_cat) + (test_pred_lgb * weight_lgb) + (test_pred_xgb * weight_xgb)
    
    # 物理兜底
    final_pred = np.clip(final_pred, 0, None).round().astype(int)
    
    # 导出
    result_df = pd.DataFrame({
        'ID' if isinstance(test_ids.iloc[0], (int, np.integer)) else 'dteday': test_ids,
        'cnt': final_pred
    })
    result_df.to_csv("ensemble_submission.csv", index=False)
    print("🎉 融合大招释放完毕！预测结果已保存至: ensemble_submission.csv")

if __name__ == "__main__":
    main()