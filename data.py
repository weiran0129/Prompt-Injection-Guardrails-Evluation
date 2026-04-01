import pandas as pd
from datasets import load_dataset

def prepare_experiment_datasets():
    print("Loading datasets from Hugging Face...")

    # --- 1. Process safe-guard-prompt-injection ---
    # 这个数据集的原始列名是 'text'
    dataset_sg = load_dataset("xTRam1/safe-guard-prompt-injection", split='train') 
    df_sg = dataset_sg.to_pandas()
    
    # 【关键修复】：将 'text' 重命名为 'prompt'
    df_sg = df_sg.rename(columns={'text': 'prompt'})
    
    # 现在这里就不会报错了
    unsafe_sg = df_sg[df_sg['label'] == 1].head(300)[['prompt']]
    unsafe_sg['source'] = 'safe-guard-injection'
    unsafe_sg['label'] = 'unsafe'

    # --- 2. Process XSTest ---
    # XSTest 里的列名本来就是 'prompt'
    dataset_xs = load_dataset("walledai/XSTest", split='test')
    df_xs = dataset_xs.to_pandas()
    
    # 提取 200 个不安全样本
    unsafe_xs = df_xs[df_xs['label'] == 'unsafe'][['prompt']]
    unsafe_xs['source'] = 'xstest'
    unsafe_xs['label'] = 'unsafe'
    
    # 提取 250 个安全样本（用于测试误拒率 FRR）
    safe_xs = df_xs[df_xs['label'] == 'safe'][['prompt']]
    safe_xs['source'] = 'xstest'
    safe_xs['label'] = 'safe'

    # --- 3. Merge ---
    # 整合 500 个恶意样本（用于测试 ASR）
    final_unsafe_df = pd.concat([unsafe_sg, unsafe_xs], ignore_index=True)
    
    # 整合 250 个良性样本（用于测试 FRR）
    final_safe_df = safe_xs.copy()

    # --- 4. Store Locally ---
    final_unsafe_df.to_csv("dataset_unsafe_500.csv", index=False, encoding='utf-8-sig')
    final_safe_df.to_csv("dataset_safe_250.csv", index=False, encoding='utf-8-sig')

    print(f"Finished Processing")
    print(f"Saved Unsafe Dataset: dataset_unsafe_500.csv ({len(final_unsafe_df)} rows)")
    print(f"Saved Safe Dataset: dataset_safe_250.csv ({len(final_safe_df)} rows)")

if __name__ == "__main__":
    prepare_experiment_datasets()