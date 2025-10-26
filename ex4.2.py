import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import sqlite3
from nltk.downloader import Downloader
import contractions
from cryptography.fernet import Fernet
import json
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import html

def main():
    # Download necessary NLTK data, without these the below functions wouldn't work
    resources = {
      'punkt': 'tokenizers/punkt',
      'stopwords': 'corpora/stopwords',
      'wordnet': 'corpora/wordnet',
      'vader_lexicon': 'sentiment/vader_lexicon'
    }
    d = Downloader()
    missing = [pkg for pkg in resources.keys() if not d.is_installed(pkg)]
    if missing:
        print("Missing NLTK resources (will download):", missing)
        for pkg in missing:
            d.download(pkg, quiet=True)
        print("Downloaded missing NLTK resources.")
    else:
        print("All required NLTK resources are already installed.")

    # -----------------------------------------------
    # 1. Data Preprocessing
    # -----------------------------------------------
    # 1.1. Load dataset
    def load_db():
        DB_FILE = "database.sqlite"
        try:
            conn = sqlite3.connect(DB_FILE)
            print("SQLite Database connection successful")
        except Exception as e:
            print(f"Error: '{e}'")
        posts_query = "SELECT id, content FROM Posts"
        posts = pd.read_sql_query(posts_query, conn)
        comments_query = "SELECT id, post_id, content FROM Comments"
        comments = pd.read_sql_query(comments_query, conn)
        print("- Number of posts:", len(posts))
        print("- Number of comments:", len(comments))
        conn.close()
        return posts, comments
    posts, comments = load_db()

    # 1.2. Cleaning data
    # 1.2.1. Cleaning data 1 (Remove duplicates from spammers)
    posts = posts.drop_duplicates(subset=["content"])
    print("- Number of posts after removing duplicates:", len(posts))
    comments = comments.drop_duplicates(subset=["content"])
    print("- Number of comments after removing duplicates:", len(comments))

    # 1.2.2. Clean data 2 (for each post/comment)
    def clean_text_light(text):
        # Remove URLs
        URL_PATTERN = r'\b(https?://\S+|www\.\S+)|[a-z0-9-]+(\.|\[\.\])[a-z]{2,4}/?[a-z0-9-]*\b'
        text = re.sub(URL_PATTERN, '', text, flags=re.IGNORECASE)

        # Remove mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove HTML tags & decode HTML entities
        text = re.sub(r'<.*?>', '', text)
        text = html.unescape(text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text
    
    posts['content'] = posts['content'].apply(clean_text_light)
    comments['content'] = comments['content'].apply(clean_text_light)

    # -----------------------------------------------
    # 2. Sentiment Analysis for posts and comments (VADER - no tokenization needed! - like exercise 15)
    # -----------------------------------------------
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()

    # Compute compound sentiment score for posts
    posts['sentiment_score'] = posts['content'].apply(lambda text: sia.polarity_scores(text)['compound'])
    # Compute compound sentiment score for comments
    comments['sentiment_score'] = comments['content'].apply(lambda text: sia.polarity_scores(text)['compound'])

    # Results
    print("\n\n====== Sentiment Analysis for posts and comments ======")
    print("--- Posts Sentiment Scores (first 10 demo) ---")
    print(posts[['id', 'content', 'sentiment_score']].head(10))

    print("\n--- Comments Sentiment Scores (first 10 demo) ---")
    print(comments[['id', 'content', 'sentiment_score']].head(10))
    
    print("\n--- Basic Statistics ---")
    print(f"Posts sentiment_score: mean {posts['sentiment_score'].mean():.4f}, std {posts['sentiment_score'].std():.4f}, min {posts['sentiment_score'].min():.4f}, max {posts['sentiment_score'].max():.4f}")
    print(f"Comments sentiment_score: mean {comments['sentiment_score'].mean():.4f}, std {comments['sentiment_score'].std():.4f}, min {comments['sentiment_score'].min():.4f}, max {comments['sentiment_score'].max():.4f}")

    # -----------------------------------------------
    # 3. Overall Tone Analysis
    # -----------------------------------------------
    print("\n\n====== Overall Platform Tone ======")

    def get_overall_tone(sentiment_score): # VADER thresholds from Hutto & Gilbert
        if sentiment_score > 0.05: return 'positive'
        elif sentiment_score < -0.05: return 'negative'
        else: return 'neutral'

    # Add overall tone column
    posts['overall_tone'] = posts['sentiment_score'].apply(get_overall_tone)
    comments['overall_tone'] = comments['sentiment_score'].apply(get_overall_tone)

    # Aggregate: count and mean sentiment_score per overall_tone
    print("--- Posts Overall Tone Statistics ---")
    posts_tone_stats = posts.groupby('overall_tone')['sentiment_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    posts_tone_stats['percent %'] = round(posts_tone_stats['count'] / len(posts) * 100, 2)
    posts_tone_stats = posts_tone_stats[['count', 'percent %', 'mean', 'std', 'min', 'max']]
    print(posts_tone_stats.sort_values(by='count', ascending=False))  # Sort by count (highest first)

    print("\n--- Comments Overall Tone Statistics ---")
    comments_tone_stats = comments.groupby('overall_tone')['sentiment_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    comments_tone_stats['percent %'] = round(comments_tone_stats['count'] / len(comments) * 100, 2)
    comments_tone_stats = comments_tone_stats[['count', 'percent %', 'mean', 'std', 'min', 'max']]
    print(comments_tone_stats.sort_values(by='count', ascending=False))  # Sort by count (highest first)

if __name__ == '__main__':
    main()
