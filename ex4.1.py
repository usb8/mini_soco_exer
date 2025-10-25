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


def main():
    # Download necessary NLTK data, without these the below functions wouldn't work
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    resources = {
      'punkt': 'tokenizers/punkt',
      'stopwords': 'corpora/stopwords',
      'wordnet': 'corpora/wordnet',
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

    ENCRYPTED_FILE_PATH = 'censorship.dat'
    fernet = Fernet('xpplx11wZUibz0E8tV8Z9mf-wwggzSrc21uQ17Qq2gg=')
    with open(ENCRYPTED_FILE_PATH, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    MODERATION_CONFIG = json.loads(decrypted_data)
    TIER1_WORDS = MODERATION_CONFIG['categories']['tier1_severe_violations']['words']
    TIER2_PHRASES = MODERATION_CONFIG['categories']['tier2_spam_scams']['phrases']
    TIER3_WORDS = MODERATION_CONFIG['categories']['tier3_mild_profanity']['words']

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
        data = pd.read_sql_query(posts_query, conn)
        # print(data.head())
        print("- Number of posts:", len(data))
        conn.close()
        return data
    data = load_db()

    # 1.2. Cleaning data
    # 1.2.1. Cleaning data 1 (Remove duplicate posts from spammers)
    data = data.drop_duplicates(subset=["content"])
    print("- Number of posts after removing duplicates:", len(data))

    # 1.2.2. Clean data 2 for each post
    # def clean_text(text):
    #     # Remove Tier 1 violations (whole text)
    #     TIER1_PATTERN = r'\b(' + '|'.join(TIER1_WORDS) + r')\b'
    #     if re.search(TIER1_PATTERN, text, flags=re.IGNORECASE):
    #         return None

    #     # Remove Tier 2 violations (whole text)
    #     for phrase in TIER2_PHRASES:
    #         if re.search(re.escape(phrase), text, flags=re.IGNORECASE):
    #             return None

    #     # Remove Tier 3 words from text
    #     TIER3_PATTERN = r'\b(' + '|'.join(TIER3_WORDS) + r')\b'
    #     # tier3_matches = re.findall(TIER3_PATTERN, text, flags=re.IGNORECASE)
    #     # if tier3_matches:
    #     #     print("Original text:", text)
    #     #     print("---Found Tier 3 words to remove:", tier3_matches)
    #     #     text = re.sub(TIER3_PATTERN, '', text, flags=re.IGNORECASE)
    #     #     print("Cleaned text:", text)
    #     text = re.sub(TIER3_PATTERN, '', text, flags=re.IGNORECASE)

    #     # Remove URLs
    #     URL_PATTERN = r'\b(https?://\S+|www\.\S+)|[a-z0-9-]+(\.|\[\.\])[a-z]{2,4}/?[a-z0-9-]*\b'
    #     text = re.sub(URL_PATTERN, '', text, flags=re.IGNORECASE)

    #     return text

    # 1.3. Tokenisation, POS tagging & lemmatisation, stopword removal
    # 1.3.1. Prepare stopwords list
    stop_words = stopwords.words('english')
    # Add extra Stopwords from Exercise 14 (impact LDA).
    stop_words.extend(['would', 'good', 'always', 'amazing', 'buy', 'quick', 'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',  'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])
    # Add extra Stopwords from Zipf's law in 1.3.3
    stop_words.extend(['attend', 'watch', 'spend', 'listen' , 'discover', 'night', 'weekend','like', 'feel', 'get', 'try', 'finish', 'see', 'wait', 'change', 'find', 'take', 'read', 'let', 'never', 'keep', 'come', 'learn', 'give', 'run', 'bring', 'enter', 'hit', 'talk', 'say', 'turn', 'push', 'open', 'lose', 'visit', 'miss', 'finally', 'else', 'still', 'truly', 'nothing', 'next', 'actually', 'damn', 'shit', 'fuck', 'seriously', 'tonight', 'bit', 'everything', 'another', 'something', 'home', 'day', 'morning', 'hour', 'first', 'last', 'next', 'today', 'everyone', 'tip', 'way', 'little', 'hard', 'real', 'perfect', 'great', 'worth'])
    # Remove duplicates
    stop_words = set(stop_words)
    
    # 1.3.2. Tokenisation, POS tagging & lemmatisation, stopword removal
    def get_wordnet_pos(tag):
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN  # or None
    
    lemmatizer = WordNetLemmatizer()
    bow_list = []  # bag-of-words list  # Type: [['token1_post1', 'token2_post1'], ['token3_post2', 'token4_post2']]
    for _, row in data.iterrows():
        text = row['content']
        # text = contractions.fix(row['content'])

        # text = clean_text(text)
        # if text is None:
        #     continue
        tokens = word_tokenize(text.lower())  # tokenise each post
       
        # Problem without POS: WordNetLemmatizer treats words as nouns by default. Many verbs and adjectives will not be transformed to their base form without knowing the word POS.
        _tokens_pos_tags = nltk.pos_tag(tokens)
        # print(_tokens_pos_tags)
        tokens = [lemmatizer.lemmatize(t, pos=get_wordnet_pos(pos)) for t, pos in _tokens_pos_tags]  # lemmatise (impact LDA)
        
        tokens = [t for t in tokens if len(t) > 2]  # drop tokens shorter than 3 chars (impact LDA)
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]  # remove non-alphabetic tokens such as num, emoji, URLs, mentions, html tags, punctuation and stopwords (impact LDA)
        if len(tokens) > 0:
            bow_list.append(tokens)

    # # 1.3.3. Zipf's Law: Most frequent words are stopwords (e.g., 'the', 'is', 'in', 'and', etc.)
    # _bow_list = [t for b in bow_list for t in b]
    # word_freq = Counter(_bow_list)
    # print(word_freq.most_common(200))  # print the 200 most common words

    # -----------------------------------------------
    # 2. Create the gensim dictionary and corpus
    # -----------------------------------------------
    dictionary = Dictionary(bow_list)  # map between words and their integer ids
    # print(dictionary)
    # print(dictionary.token2id)
    # print(dictionary.token2id)
    # print(dictionary[0])
    dictionary.filter_extremes(no_below=2, no_above=0.3)  # remove very rare words (no_below) and very common (no_above) (impact LDA)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]  # Type: BOW type ->  [[(id, count), ...], ...]
    # print(corpus)

    # -----------------------------------------------
    # 3. Model selection: We don't know how many topics are there/best
    #    -> Train LDA algorithm for a range of topic counts and pick the model with best coherence score
    #    coherence using c_v is a common automatic metric for topic quality.
    # -----------------------------------------------
    best_coherence_score = -100
    best_lda_model = None
    lda_model_k_10 = None
    best_num_topics = 0
    for K in range(2, 11):
        # Train an LDA model (impact LDA) with:
        #   K topics
        #   the algorithm will iterate over the corpus 10 times (larger number -> slower but can help convergence).
        #   random_state (refer ML or DataMining textbooks)
        lda_model = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2)

        # Then compute the coherence score for lda_model
        coherence_model = CoherenceModel(model=lda_model, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        if coherence_score > best_coherence_score:
            print(f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!')
            best_coherence_score = coherence_score
            best_lda_model = lda_model
            best_num_topics = K
        else: 
            print(f'Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.')
        
        if K == 10:
            lda_model_k_10 = lda_model

    # -----------------------------------------------
    # 4. Show topics of the chosen model
    # -----------------------------------------------
    # Print top 5 most representative words per topic
    print(f'These are the words most representative of each of the {best_num_topics} topics:')
    for i, topic in best_lda_model.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")

    print('For ex.4.1======================================')
    print('These are the topics of the LDA model with K=10:')
    for i, topic in lda_model_k_10.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")
    print('================================================')

    # -----------------------------------------------
    # 5. Count dominant topic per document
    # -----------------------------------------------
    # Count the dominant topic for each document
    topic_counts = [0] * best_num_topics  # one counter per topic
    for bow in corpus:
        topic_dist = best_lda_model.get_document_topics(bow)  # list of (topic_id, probability)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]  # find the top probability
        topic_counts[dominant_topic] += 1  # add 1 to the most probable topic's counter
    # Display the topic counts
    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")


if __name__ == '__main__':
    main()
