import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the classified data
df_to_analyze = pd.read_csv('/Users/kimberleyliao/Desktop/capstone/Competitors/competitors_lda_classified.csv')

print("\nSentiment Distribution:")
print(df_to_analyze['sentiment'].value_counts())

def analyze_topics_for_sentiment(texts, n_topics=5, n_words=10, n_samples=10):
    # Create document-term matrix
    # Create vectorizer with custom token pattern to ignore 'https'
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2, 
        stop_words='english',
        token_pattern=r'(?u)(?!https)\b\w\w+\b'
    )
    dtm = vectorizer.fit_transform(texts)
    
    # Apply LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )
    
    # Get topic distribution
    # Fit LDA model and get topic distribution for each document
    doc_topics = lda.fit_transform(dtm)
    # For each document, get the topic with highest probability, This gives us the most likely topic for each text
    dominant_topics = doc_topics.argmax(axis=1)
    # Create a temporary dataframe for this sentiment's texts
    temp_df = pd.DataFrame({'text': texts, 'topic': dominant_topics})
    
    # Print topic distribution for this sentiment
    print("\nTopic distribution:")
    print(temp_df['topic'].value_counts(normalize=True).round(3))
    
    # For each topic
    for topic_idx in range(n_topics):
        print(f"\n--- Topic {topic_idx + 1} ---")
        
        # Get top words
        top_words_idx = lda.components_[topic_idx].argsort()[-n_words:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        print(f"Top words: {', '.join(top_words)}")
        
        # Get sample tweets for this topic
        topic_mask = (topic_idx == doc_topics.argmax(axis=1))  # Get tweets where this is dominant topic
        topic_indices = np.where(topic_mask)[0]
        
        # Get view counts and indices for tweets in this topic
        topic_tweets = [(i, df_to_analyze['view_count'].iloc[i]) 
                       for i in topic_indices]
        
        # Sort by view count and get top n_samples
        sorted_tweets = sorted(topic_tweets, key=lambda x: x[1], reverse=True)[:n_samples]
        
        print(f"\nTop {n_samples} most viewed tweets for this topic:")
        for idx, views in sorted_tweets:
            print(f"\nTweet (Views: {views}):")
            print(texts[idx])
            print("-" * 80)


        # # Print AI-related tweets for this topic
        # print(f"\nAI-related tweets for this topic:")
        # ai_count = 0
        # for idx in topic_tweets_idx:
        #     if 'ai' in texts[idx].lower() or 'artificial intelligence' in texts[idx].lower():
        #         topic_prob = doc_topics[idx][topic_idx]
        #         print(f"\nTweet (Topic Probability: {topic_prob:.3f}):")
        #         print(texts[idx])
        #         print("-" * 80)
        #         ai_count += 1
        # if ai_count == 0:
        #     print("No AI-related tweets found in this topic's samples")
                    
# Analyze topics for each sentiment
for sentiment in df_to_analyze['sentiment'].unique():
    print(f"\n{'='*50}")
    print(f"ANALYZING SENTIMENT: {sentiment}")
    print(f"{'='*50}")
    
    # Get tweets for this sentiment
    sentiment_mask = df_to_analyze['sentiment'] == sentiment
    sentiment_tweets = df_to_analyze[sentiment_mask]['text'].dropna().tolist()
    
    print(f"Number of tweets: {len(sentiment_tweets)}")
    
    if len(sentiment_tweets) < 5:
        print(f"Not enough tweets for sentiment: {sentiment}")
        continue
        
    try:
        # Create document-term matrix and LDA model to get topic distributions
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(sentiment_tweets)
        lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
        doc_topics = lda.fit_transform(dtm)
        
        # Count tweets per topic (using highest probability topic for each tweet)
        topic_assignments = doc_topics.argmax(axis=1)
        tweets_per_topic = np.bincount(topic_assignments)
        
        print("\nTweets per topic:")
        for topic_idx, count in enumerate(tweets_per_topic):
            print(f"Topic {topic_idx + 1}: {count} tweets")
            
        analyze_topics_for_sentiment(sentiment_tweets)
    except Exception as e:
        print(f"Error analyzing sentiment {sentiment}: {str(e)}")

        # Create PDF report
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from io import StringIO
        import sys

        # Redirect stdout to capture print output
        old_stdout = sys.stdout
        output = StringIO()
        sys.stdout = output

        # Re-run the analysis to capture output
        for sentiment in df_to_analyze['sentiment'].unique():
            print(f"\n{'='*50}")
            print(f"ANALYZING SENTIMENT: {sentiment}")
            print(f"{'='*50}")
            
            sentiment_mask = df_to_analyze['sentiment'] == sentiment
            sentiment_tweets = df_to_analyze[sentiment_mask]['text'].dropna().tolist()
            
            print(f"Number of tweets: {len(sentiment_tweets)}")
            
            if len(sentiment_tweets) < 5:
                print(f"Not enough tweets for sentiment: {sentiment}")
                continue
                
            try:
                vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                dtm = vectorizer.fit_transform(sentiment_tweets)
                lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
                doc_topics = lda.fit_transform(dtm)
                
                topic_assignments = doc_topics.argmax(axis=1)
                tweets_per_topic = np.bincount(topic_assignments)
                
                print("\nTweets per topic:")
                for topic_idx, count in enumerate(tweets_per_topic):
                    print(f"Topic {topic_idx + 1}: {count} tweets")
                    
                analyze_topics_for_sentiment(sentiment_tweets)
            except Exception as e:
                print(f"Error analyzing sentiment {sentiment}: {str(e)}")

        # Get the output
        analysis_text = output.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout

        # Create PDF
        doc = SimpleDocTemplate("sentiment_topic_analysis.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom style for code/output blocks
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=8,
            leading=10,
            leftIndent=20
        )
        
        # Build PDF content
        content = []
        
        # Add title
        content.append(Paragraph("Sentiment and Topic Analysis Report", styles['Title']))
        content.append(Spacer(1, 12))
        
        # Split analysis text into sections and add to PDF
        sections = analysis_text.split("\n\n")
        for section in sections:
            if section.strip():
                content.append(Paragraph(section.replace("\n", "<br/>"), code_style))
                content.append(Spacer(1, 12))
                
        # Generate PDF
        doc.build(content)
        print("Analysis saved to sentiment_topic_analysis.pdf")
