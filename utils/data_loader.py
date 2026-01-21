import pandas as pd


def load_dataset(filepath, target_column):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The File '{filepath}' was not found")
        return None, None

    X_df = df.drop(columns=[target_column])
    y_series = df[target_column]

    for col in X_df.columns:
        if X_df[col].dtype == "object":
            print(f"Converting column '{col}' from text to numbers...")
            X_df[col] = X_df[col].astype("category").cat.codes

    if y_series.dtype == "object":
        y_series = y_series.astype("category").cat.codes

    X = X_df.values
    y = y_series.values

    print(f"Successfully loaded dataset: {X.shape[0]} rows, {X.shape[1]} features.")
    return X, y
