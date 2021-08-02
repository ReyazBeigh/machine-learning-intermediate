import pandas as pd

csv_path = "./input/melb_data.csv"

pd_data = pd.read_csv(csv_path)

print(pd_data.head())

