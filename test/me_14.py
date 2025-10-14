import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

def main():
    # Download necessary NLTK data, without these the below functions wouldn't work
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load dataset
    data = pd.read_csv("./test/posts.csv")

    # 1.a. Stopwords
    stop_words = stopwords.words('english')

    # 1.b. Add extra Stopwords (impact LDA).
    stop_words.extend(['would', 'best', 'always', 'amazing', 'bought', 'quick' 'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 'well',  'life', 'said', 'year', 'going', 'good', 'really', 'much', 'want', 'back', 'look', 'article', 'host', 'university', 'reply', 'thanks', 'mail', 'post', 'please'])

    # 1.c. Tokenisation, lemmatisation and filtering pipeline
    lemmatizer = WordNetLemmatizer()
    bow_list = []  # bag-of-words list  # Type: [['token1_post1', 'token2_post1'], ['token3_post2', 'token4_post2']]
    for _, row in data.iterrows():
        text = row['content']
        tokens = word_tokenize(text.lower())  # tokenise each post
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatise (impact LDA)
        tokens = [t for t in tokens if len(t) > 2]  # drop tokens shorter than 3 chars (impact LDA)
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]  # remove non-alphabetic tokens and stopwords (impact LDA)
        if len(tokens) > 0:
            bow_list.append(tokens)

    # 2. Create the gensim dictionary and corpus
    dictionary = Dictionary(bow_list)  # map between words and their integer ids
    # print(dictionary)
    # print(dictionary.token2id)
    # print(dictionary.token2id)
    # print(dictionary[0])
    dictionary.filter_extremes(no_below=2, no_above=0.3)  # remove very rare words (no_below) and very common (no_above) (impact LDA)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]  # Type: BOW type ->  [[(id, count), ...], ...]
    # print(corpus)

    # 3. Model selection: We don't know how many topics are there/best
    #    -> Train LDA algorithm for a range of topic counts and pick the model with best coherence score
    #    coherence using c_v is a common automatic metric for topic quality.
    best_coherence_score = -100
    best_lda_model = None
    best_num_topics = 0
    for K in range(2, 10):
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

    # 4. Show topics of the chosen model
    # Print top 5 most representative words per topic
    print(f'These are the words most representative of each of the {best_num_topics} topics:')
    for i, topic in best_lda_model.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")

    # 5. Count dominant topic per document
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
