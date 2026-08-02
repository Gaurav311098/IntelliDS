import pandas as pd


def read_dataset(file):
    """
    Reads uploaded CSV/Excel file.
    Returns pandas DataFrame.
    """

    filename = file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(file)

    if filename.endswith(".xls"):
        return pd.read_excel(file)

    raise ValueError("Unsupported file format.")