import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def run_inference(model, data_path, dataset_name, output_dir, master_label_df=None):
    """
    通用推理函数 (支持通过ID匹配外部标签)
    :param model: 已加载的模型对象
    :param data_path: 数据CSV路径
    :param dataset_name: 数据集名称
    :param output_dir: 结果保存目录
    :param master_label_df: 包含所有ID和Label的总表 DataFrame (可选)
    """
    print(f"\n{'='*20} 正在处理: {dataset_name} {'='*20}")
    print(f"数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到文件 {data_path}，跳过该数据集。")
        return None

    # 1. 加载特征数据
    df = pd.read_csv(data_path)
    
    # === 关键修改：统一 ID 格式为字符串，防止匹配失败 ===
    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str)
    else:
        print("⚠️ 警告: 数据中没有 'ID' 列，无法匹配标签，将仅进行预测。")
        ids = range(len(df)) # 生成默认索引
        # 如果没有ID列，无法进行 merge，只能用 df 自身
        master_label_df = None 

    # 2. 匹配标签 (Label Matching)
    y_true = None
    
    # 如果提供了总标签表，并且数据中有ID，则进行合并
    if master_label_df is not None and 'ID' in df.columns:
        print("正在通过 ID 匹配标签...")
        
        # 先删除特征文件中可能存在的旧 label 列，避免 merge 产生 label_x, label_y
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
            
        # Left Join: 保留特征文件的所有行，匹配上的填入标签，没匹配上的为 NaN
        df_merged = pd.merge(df, master_label_df[['ID', 'label']], on='ID', how='left')
        
        # 提取标签
        if 'label' in df_merged.columns:
            y_true = df_merged['label'].values
            
            # 检查是否有匹配失败的情况
            nan_count = df_merged['label'].isna().sum()
            if nan_count > 0:
                print(f"⚠️ 注意: 有 {nan_count} 个样本未在 label.csv 中找到对应的 ID (Label 为 NaN)。")
        
        # 更新用于预测的 df (此时 df_merged 包含特征 + ID + label)
        # 我们需要把 ID 和 label 拿走，只留特征
        ids = df_merged['ID']
        data_for_pred = df_merged.drop(columns=['ID', 'label'])
        
    else:
        # 如果没有提供总表，尝试直接从文件读取 label
        if 'ID' in df.columns:
            ids = df['ID']
            data_for_pred = df.drop(columns=['ID'])
        else:
            ids = range(len(df))
            data_for_pred = df.copy()
            
        if 'label' in data_for_pred.columns:
            y_true = data_for_pred['label'].values
            data_for_pred = data_for_pred.drop(columns=['label'])

    # 3. 特征对齐 (自动筛选逻辑)
    expected_features = getattr(model, "n_features_in_", None)
    
    if expected_features and data_for_pred.shape[1] > expected_features:
        print(f"ℹ️ 检测到输入特征数 ({data_for_pred.shape[1]}) 多于模型需求 ({expected_features})")
        print("尝试加载 'train_selected_features_lasso.csv' 获取特征名称列表以进行筛选...")
        
        # 寻找参考文件
        feature_ref_path = os.path.join(os.path.dirname(data_path), 'train_selected_features_lasso.csv')
        # 如果当前目录下找不到，也可以尝试写死一个路径或者从 data_path 推导
        if not os.path.exists(feature_ref_path):
             # 尝试在上级目录找 (根据你的文件结构可能需要调整)
             feature_ref_path = r"F:\new_yq\data\yq5mm\exval_data\img\train_selected_features_lasso.csv"

        if os.path.exists(feature_ref_path):
            df_ref = pd.read_csv(feature_ref_path)
            feature_names = [c for c in df_ref.columns if c not in ['ID', 'label']]
            
            missing_cols = [c for c in feature_names if c not in data_for_pred.columns]
            if not missing_cols:
                print(f"✅ 成功匹配特征列表，筛选出 {len(feature_names)} 个特征。")
                data_for_pred = data_for_pred[feature_names]
            else:
                print(f"❌ 错误: 缺失特征: {missing_cols}")
                return None
        else:
            print(f"⚠️ 警告: 找不到特征参考文件: {feature_ref_path}")
    
    # 4. 预测
    try:
        pred_labels = model.predict(data_for_pred)
        
        if hasattr(model, "predict_proba"):
            pred_probs = model.predict_proba(data_for_pred)[:, 1]
        else:
            d_vals = model.decision_function(data_for_pred)
            pred_probs = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min())
            
    except Exception as e:
        print(f"❌ 预测错误: {e}")
        return None

    # 5. 计算指标 & 输出
    auc_score = "N/A"
    
    if y_true is not None:
        # ⚠️ 关键：计算指标时要移除 NaN 的 Label (未匹配到的数据)
        valid_mask = ~pd.isna(y_true)
        if np.sum(valid_mask) > 0:
            # 仅在有标签的数据上计算 AUC
            y_true_valid = y_true[valid_mask]
            pred_probs_valid = pred_probs[valid_mask]
            pred_labels_valid = pred_labels[valid_mask]
            
            try:
                auc_score = roc_auc_score(y_true_valid, pred_probs_valid)
                acc_score = accuracy_score(y_true_valid, pred_labels_valid)
                print(f"📊 {dataset_name} 指标 (基于 {np.sum(valid_mask)} 个匹配样本):")
                print(f"   AUC:      {auc_score:.4f}")
                print(f"   Accuracy: {acc_score:.4f}")
            except ValueError as e:
                print(f"⚠️ 无法计算指标 (可能是标签只有一个类别): {e}")
        else:
            print("⚠️ 所有样本均未匹配到标签，无法计算指标。")

    # 6. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    res_df = pd.DataFrame()
    res_df['ID'] = ids
    if y_true is not None:
        res_df['True_Label'] = y_true
    res_df['SVM_Predicted'] = pred_labels
    res_df['SVM_Probability'] = pred_probs
    
    save_path = os.path.join(output_dir, f'inference_result_{dataset_name}.csv')
    res_df.to_csv(save_path, index=False)
    print(f"💾 结果已保存: {save_path}")
    
    return auc_score


if __name__ == "__main__":
    # ================= 配置区域 =================
    
    model_path = r"model\in\SVM.pkl"
    # 总 Label 表的路径
    label_path = r"data\label.csv"
    
    output_directory = r"result\in"
    
    datasets_to_process = {
        # 你的数据集路径
        "train": r"data\in\train_selected_features_lasso.csv",
        "test": r"data\in\train_selected_features_lasso.csv",
        # "ExVal_1": r"F:\new_yq\data\yq5mm\exval_data1\step8.0_RadiomicsCombind\deleteNaN.csv",
    }
    
    # ================= 执行区域 =================
    
    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            svm_model = pickle.load(f)
    else:
        print("错误：找不到模型文件！")
        exit()

    # 2. === 新增步骤：加载总 Label 表 ===
    print(f"正在加载 Label 表: {label_path}")
    master_label_df = None
    if os.path.exists(label_path):
        master_label_df = pd.read_csv(label_path)
        # 强制将 ID 转为 string，确保与特征表中的 ID 类型一致，否则 merge 会失败
        if 'ID' in master_label_df.columns:
            master_label_df['ID'] = master_label_df['ID'].astype(str)
            print(f"Label 表加载成功，包含 {len(master_label_df)} 个样本。")
        else:
            print("❌ 错误：Label 表中找不到 'ID' 列！")
            exit()
    else:
        print("⚠️ 警告：找不到 Label 表文件，后续将无法计算 AUC 指标。")

    # 3. 循环处理
    summary_results = []
    
    for name, path in datasets_to_process.items():
        if not path or not os.path.exists(path):
            print(f"跳过 {name}: 路径不存在")
            continue
            
        # 将 master_label_df 传递给函数
        auc = run_inference(svm_model, path, name, output_directory, master_label_df)
        summary_results.append({"Dataset": name, "AUC": auc})

    # 4. 汇总
    print("\n" + "="*30)
    print("       最终结果汇总       ")
    print("="*30)
    df_summary = pd.DataFrame(summary_results)
    print(df_summary)
    
    if not df_summary.empty:
        df_summary.to_csv(os.path.join(output_directory, "summary_auc.csv"), index=False)