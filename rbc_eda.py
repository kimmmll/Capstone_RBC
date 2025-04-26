import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import softmax
import urllib.request
import csv
from tqdm import tqdm

import re


df_RBC= pd.read_csv('/Users/kimberleyliao/Desktop/capstone/rbc_all_keywords/rbc_full_tweets.csv')
df_RBC = df_RBC.drop_duplicates(subset=['text'])
df_RBC.head()
df_RBC.info()

# Create a new column 'contains_rbc' that checks if 'rbc' exists in text (case insensitive)
df_RBC = df_RBC.drop_duplicates(subset=['text'])
df_RBC['contains_rbc'] = df_RBC['text'].str.lower().str.contains('rbc')

# Display count of tweets containing 'rbc'
print("\nNumber of tweets containing 'rbc':", df_RBC['contains_rbc'].sum())
print("Percentage of tweets containing 'rbc': {:.2f}%".format(
    (df_RBC['contains_rbc'].sum() / len(df_RBC)) * 100
))

#Top 10 Most Active Users by Brand
#Potential Official Accounts: Future analysis remove official profiles that lead to a higher neutral clasification)
def top_active_users(df, top_n=10):
    return df['author_username'].value_counts().head(top_n)

print(top_active_users(df_RBC))

# Total Number of Posts/Tweets by Year for RBC
# First extract year from created_time
df_RBC['year'] = pd.to_datetime(df_RBC['created_time']).dt.year
tweets_per_year = df_RBC['year'].value_counts().sort_index()
print("\nNumber of posts/tweets by year for RBC:\n", tweets_per_year)

# Plot
tweets_per_year.plot(kind='bar', figsize=(8, 5), title='RBC Posts/Tweets Over Years', color='lightgreen')
plt.ylabel('Number of Posts/Tweets')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
