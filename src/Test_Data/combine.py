import pandas as pd

# Load both datasets
safe_df = pd.read_csv("./dataset_safe_250.csv")
unsafe_df = pd.read_csv("./dataset_unsafe_500.csv")

# Combine
df = pd.concat([safe_df, unsafe_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("combined_shuffled.csv", index=False)

print("Saved combined_shuffled.csv")
