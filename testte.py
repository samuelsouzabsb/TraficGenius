import pandas as pd

df = pd.read_parquet(r"C:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_consolidado.parquet")

print(df.head())