import pandas as pd

df_competitors = pd.read_csv('/Users/kimberleyliao/Desktop/capstone/Competitors/merged_competitors.csv')

df_competitors = df_competitors.drop_duplicates(subset=['text'])


df_competitors.to_csv('/Users/kimberleyliao/Desktop/capstone/Competitors/competitors_data_clean.csv', index=False)
df_competitors.describe()
df_competitors.info()
df_competitors.isnull().sum()
print(df_competitors.shape)

