import ssl
import urllib.request

# Add SSL certificate verification bypass
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import urllib.request
import csv
from tqdm import tqdm

import re

df_competitors = pd.read_csv('/Users/kimberleyliao/Desktop/capstone/Competitors/competitors_data_clean.csv')
df_to_analyze = df_competitors
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

print("\n Loaded Sentiment Labels:", labels)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Sentiment analysis function
def get_sentiment(text):
    try:
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]  # descending order
        sentiment_label = labels[ranking[0]]  # pick the label with highest score
        sentiment_score = scores[ranking[0]]  # probability score
        return sentiment_label, float(sentiment_score)
    except Exception as e:
        print("Error in processing:", e)
        return None, None
    
# Apply RoBERTa to the dataset


# Prepare lists to store results
sentiments = []
confidence_scores = []

# Loop over each text and classify
print("\n Running sentiment analysis\n")
for text in tqdm(df_to_analyze['text'], desc="Analyzing Sentiment"):
    sentiment, score = get_sentiment(text)
    sentiments.append(sentiment)
    confidence_scores.append(score)

# Add results to dataframe
df_to_analyze['sentiment'] = sentiments
df_to_analyze['sentiment_score'] = confidence_scores

print("\n Sentiment analysis completed!")

# Sentiment distribution
print("\n Sentiment Distribution (%):")
print(df_to_analyze['sentiment'].value_counts(normalize=True) * 100)

# # Sample 50 neutral tweets to review why they were classified as neutral
# neutral_sample = df_to_analyze[df_to_analyze['sentiment'] == 'neutral'].sample(50, random_state=42)
# print("\n Sample of Neutral Tweets:\n")
# print(neutral_sample[['text', 'author_username']])

# LDA for each segment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Prepare data
texts = df_to_analyze['text'].dropna().tolist()

# Convert to document-term matrix
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(texts)

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics
lda.fit(dtm)

# Get topic distribution for each document (tweet)
topic_distribution = lda.transform(dtm)

# Assign the dominant topic (highest probability) for each tweet
dominant_topic = topic_distribution.argmax(axis=1)

# Add this as a new column
df_to_analyze['topic'] = dominant_topic

# Check if assigned correctly
print(df_to_analyze[['text', 'topic']].head())

# Display topics
for index, topic in enumerate(lda.components_):
    print(f"\n💡 Topic #{index}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])  # Top 10 words in topic
    
    # Get documents most representative of this topic
    topic_mask = (df_to_analyze['topic'] == index)
    topic_docs = df_to_analyze[topic_mask].sort_values('view_count', ascending=False)
    top_3_docs = topic_docs.head(3)
    
    print("\nMost viewed tweets for this topic:")
    for i, (text, views) in enumerate(zip(top_3_docs['text'], top_3_docs['view_count']), 1):
        print(f"{i}. [Views: {views}] {text}")  # Print full text
    

topic_sentiment = df_to_analyze.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)

print("\n Sentiment Distribution by Topic:\n")
print(topic_sentiment)

# Normalize to get percentages
topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
print("\n Sentiment Distribution by Topic (%) :\n")
print(topic_sentiment_percent)

# Save the results
df_to_analyze.to_csv('competitors_lda_classified.csv', index=False)