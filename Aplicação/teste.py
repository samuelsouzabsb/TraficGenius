import pandas as pd
df = pd.read_parquet(r"C:\Users\samuelbarroso\Documents\Desenvolvimento\TraficGenius\dataset\dataset_amostra_limpa_avancado.parquet")


df.info()

df.head()
print(df)