import pandas as pd

REQUIRED_COLUMNS = ["prompt", "label"]


def load_dataset(csv_path):
    """
    Expects a CSV with columns: prompt, source, label
    label must be either 'safe' or 'unsafe'
    """
    df = pd.read_csv(csv_path)

    # Normalize column names (handle BOM / hidden chars)
    df.columns = [c.strip().lower().replace("\ufeff", "") for c in df.columns]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[REQUIRED_COLUMNS].copy()

    # Normalize labels
    df["label"] = df["label"].str.lower().str.strip()
    if not set(df["label"]).issubset({"safe", "unsafe"}):
        raise ValueError("Labels must be 'safe' or 'unsafe'")

    return df.to_dict(orient="records")
