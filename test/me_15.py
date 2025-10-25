"""
Using Valence Aware Dictionary and sEntiment Reasoner (VADER), determine the highest and lowest rated Delifrance locations across the US based on the given review dataset reviews.csv [1].
- First, follow along the NLTK library’s documentation[2] to set up a basic workflow.
- Then, improve your classification by adding custom stopwords, removing short words and lemmatisation.
- Bonus: Plot the yearly average sentiment and determine the year with the best average review sentiment.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER lexicon required for SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


df = pd.read_csv('./test/reviews.csv')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute compound sentiment score for each review
df['sentiment_score'] = df['Review'].apply(lambda review: sia.polarity_scores(review)['compound'])

# Aggregate: compute count and mean sentiment per location
location_stats = df.groupby('location')['sentiment_score'].agg(['count', 'mean'])

# Keep only locations with at least 3 reviews
filtered_sentiment_by_location = location_stats[location_stats['count'] >= 3]

# Sort locations by mean sentiment (highest first)
sorted_filtered_sentiment = filtered_sentiment_by_location.sort_values(by='mean', ascending=False)

print("Average sentiment score for each location with at least 3 reviews:")
print(sorted_filtered_sentiment)

# ------------------------------------------------------------
# Time-series analysis: rating over time (yearly average sentiment and plotting)
# ------------------------------------------------------------

# Clean 'Date' column (strip quotes/whitespace) and convert to datetime objects/format
df['Date'] = df['Date'].str.strip().str.replace('"', '')
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

# Use Date as index to enable resampling by year
df.set_index('Date', inplace=True)

# Resample by year and compute mean sentiment per year
yearly_sentiment = df['sentiment_score'].resample('Y').mean()

# Plot yearly sentiment trend
plt.figure(figsize=(6, 4))
yearly_sentiment.plot(marker='o', linestyle='-')

# Formatting
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()
