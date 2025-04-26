import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import urllib.request
import csv
from tqdm import tqdm

import re

df_competitors= pd.read_csv('/Users/kimberleyliao/Desktop/capstone/Competitors/competitors_data_clean.csv')

df_competitors.head()
df_competitors.info()
df_to_analyze = df_competitors
# Create a new column 'contains_ubs' that checks if 'ubs' exists in text (case insensitive)
df_to_analyze = df_to_analyze.drop_duplicates(subset=['text'])
df_to_analyze['contains_ubs'] = df_to_analyze['text'].str.lower().str.contains('ubs')

# Display count of tweets containing 'ubs'
print("\nNumber of tweets containing 'ubs':", df_to_analyze['contains_ubs'].sum())
print("Percentage of tweets containing 'ubs': {:.2f}%".format(
    (df_to_analyze['contains_ubs'].sum() / len(df_to_analyze)) * 100
))

#Top 10 Most Active Users by Brand
#Potential Official Accounts: Future analysis remove official profiles that lead to a higher neutral clasification)
def top_active_users(df, top_n=10):
    return df['author_username'].value_counts().head(top_n)

print(top_active_users(df_to_analyze))

# Total Number of Posts/Tweets by Year for UBS
# First extract year from created_time
df_to_analyze['year'] = pd.to_datetime(df_to_analyze['created_time']).dt.year
tweets_per_year = df_to_analyze['year'].value_counts().sort_index()
print("\nNumber of posts/tweets by year for UBS:\n", tweets_per_year)

# Plot
tweets_per_year.plot(kind='bar', figsize=(8, 5), title='Competitors Posts/Tweets Over Years', color='lightgreen')
plt.ylabel('Number of Posts/Tweets')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Count number of replies
num_replies = len(df_to_analyze[df_to_analyze['post_type']=='reply'])
print(f"\nNumber of replies: {num_replies}")