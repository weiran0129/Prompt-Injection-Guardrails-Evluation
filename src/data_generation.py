from datasets import load_dataset
import pandas as pd

RANDOM_SEED = 42
OUTPUT_CSV = "prompt_injeciton_data.csv"


def sample_neuralchemy():
    ds = load_dataset("neuralchemy/Prompt-injection-dataset", "full")
    df = ds["train"].to_pandas()

    required_cols = {"text", "label", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Neuralchemy dataset missing columns: {missing}")

    # ================= filter <= 500 length =================
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() <= 500]
    # ================================================================

    # 1) direct_injection
    direct_df = df[df["category"] == "direct_injection"]
    if len(direct_df) < 550:
        raise ValueError(f"Not enough direct_injection samples after length filter: found {len(direct_df)}")
    direct_sample = direct_df.sample(n=30, random_state=RANDOM_SEED)

    # 2) indirect_injection
    indirect_df = df[df["category"] == "indirect_injection"]
    if len(indirect_df) < 30:
        raise ValueError(f"Not enough indirect_injection samples after length filter: found {len(indirect_df)}")
    indirect_sample = indirect_df.sample(n=5, random_state=RANDOM_SEED)

    # 3) benign
    benign_df = df[df["category"] == "benign"]
    if len(benign_df) < 200:
        raise ValueError(f"Not enough benign samples after length filter: found {len(benign_df)}")
    benign_sample = benign_df.sample(n=20, random_state=RANDOM_SEED)

    result = pd.concat([direct_sample, indirect_sample, benign_sample], ignore_index=True)
    result = result[["text", "label"]].copy()
    result["label"] = result["label"].astype(int)

    return result


def sample_jasper():
    ds = load_dataset("JasperLS/prompt-injections")

    # 合并 train + test
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    df = pd.concat([train_df, test_df], ignore_index=True)

    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Jasper dataset missing columns: {missing}")

    # ================= filter <= 500 length =================
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() <= 500]
    # ================================================================

    # Attack
    attack_df = df[df["label"] == 1]
    if len(attack_df) < 220:
        raise ValueError(f"Not enough attack samples in Jasper after length filter: found {len(attack_df)}")

    attack_sample = attack_df.sample(n=5, random_state=RANDOM_SEED).copy()
    attack_sample["label"] = attack_sample["label"].astype(int)

    return attack_sample[["text", "label"]]


def main():
    print("Loading and sampling datasets... This may take a moment.")
    neuralchemy_part = sample_neuralchemy()
    jasper_part = sample_jasper()

    combined = pd.concat([neuralchemy_part, jasper_part], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(combined)}")
    print(f"Max text length: {combined['text'].str.len().max()} characters")
    
    print("\nLabel distribution:")
    print(combined["label"].value_counts())

    print("\nPreview:")
    print(combined.head())


if __name__ == "__main__":
    main()