import pandas as pd

df = pd.read_csv('prompt_injeciton_data.csv')

df = df.rename(columns={'text': 'prompt'})


df['label'] = df['label'].map({1: 'unsafe', 0: 'safe'})


df.to_csv('test_data.csv', index=False)

print("转换完成！新文件已生成。")