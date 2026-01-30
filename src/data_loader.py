import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

print(DATA_DIR)
def fetch_data(filepath: str) -> pd.DataFrame:
    """:param filepath: filepath relative to data folder
        :rtype: DataFrame
    """

    "Fetches data from csv file"
    filepath = DATA_DIR/f'{filepath}.csv'

    if filepath.is_file():
        return pd.read_csv(filepath)
    

