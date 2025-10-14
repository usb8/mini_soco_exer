import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download the VADER lexicon
nltk.download('vader_lexicon')

df = pd.read_csv('reviews.csv')

# Initialize the VADER sentiment analyser
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each review
df['sentiment_score'] = df['Review'].apply(lambda review: sia.polarity_scores(review)['compound'])

# First, calculate both the count and mean sentiment for each location
location_stats = df.groupby('location')['sentiment_score'].agg(['count', 'mean'])

# Filter for locations with at least 3 reviews**
filtered_sentiment_by_location = location_stats[location_stats['count'] >= 3]

# Sort the filtered results by the mean sentiment score
sorted_filtered_sentiment = filtered_sentiment_by_location.sort_values(by='mean', ascending=False)

print("Average sentiment score for each location with at least 3 reviews:")
print(sorted_filtered_sentiment)

# Plot change of rating over time.

# Clean the 'Date' column by removing extra quotes and spaces and turn into datetime objects
df['Date'] = df['Date'].str.strip().str.replace('"', '')
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

# Set the 'Date' column as the index for time-series resampling
df.set_index('Date', inplace=True)

# Resample the data by year ('Y') and calculate the mean sentiment score for each year
yearly_sentiment = df['sentiment_score'].resample('Y').mean()

# Create plot
plt.figure(figsize=(6, 4))
yearly_sentiment.plot(marker='o', linestyle='-')

# Formatting
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot
plt.show()
