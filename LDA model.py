from gensim import corpora, models
from random import uniform
import Preprocess
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from sklearn import preprocessing
import re
import os
import random

import multiprocessing
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# defined stop words
cachedStopWords = stopwords.words("english")

# Load the Reuters dataset
from nltk.corpus import reuters

data_path = 'C:/Users/Ali/Downloads/reuters21578'


def load_data():

    topics_dic = {}
    bodies_dic = {}

    for filename in os.listdir(data_path):
        if filename.startswith('reut2-'):
            file_path = os.path.join(data_path, filename)
            print(file_path)
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                # Use regex to extract the text content between <BODY> and </BODY> tags
                match_documents = re.findall(r'<REUTERS([^>]*)>(.*?)</REUTERS>', content, re.DOTALL)
                print(len(match_documents))
                for doc_id, doc_content in match_documents:
                    doc_id = re.search(r'NEWID="([^"]+)"', doc_id).group(1)
                    match_topic = re.search(r'<TOPICS[^>]*>(.*?)</TOPICS>', doc_content, re.DOTALL)
                    match_body = re.search(r'<BODY[^>]*>(.*?)</BODY>', doc_content, re.DOTALL)

                    if match_body:
                        doc_topics = re.findall(r'<D[^>]*>(.*?)</D>', match_topic.group(1), re.DOTALL)
                        doc_body = re.sub(r'<[^>]+>', '', match_body.group(1))

                        topics_dic[doc_id] = doc_topics
                        bodies_dic[doc_id] = doc_body

    return topics_dic, bodies_dic


def get_average_length(tokenized_documents, num_documents):
    sublist_lengths = [len(doc) for doc in tokenized_documents]
    mode = st.mode(sublist_lengths)
    average = sum(sublist_lengths) / num_documents
    print("Average:", average, "\nMode:", mode)


def plot_length_distribution(tokenized_documents):
    sublist_lengths = [len(doc) for doc in tokenized_documents]
    # Plot the distribution of sublist lengths
    plt.hist(sublist_lengths, bins=range(min(sublist_lengths), max(sublist_lengths) + 2), edgecolor='black')
    plt.xlabel('Length of docs')
    plt.ylabel('Frequency')
    plt.title('Distribution of Documents Lengths')
    num_ticks = 10
    plt.xticks(range(min(sublist_lengths), max(sublist_lengths) + 1, (max(sublist_lengths) - min(sublist_lengths) + 1) // num_ticks))
    plt.show()


def plot_topic_distribution(categories, documents_topics, sort=False):
    topics_freq = {topic: 0 for topic in categories}
    for topic_list in documents_topics:
        for topic in topic_list:
            topics_freq[categories[topic]] += 1

    if sort:
        plt.title('Distribution of Topics - Sorted')
        topics_freq = dict(sorted(topics_freq.items(), key=lambda x: x[1], reverse=True))
    else:
        plt.title('Distribution of Topics')

    # Plot the distribution of sublist lengths
    plt.bar(list(topics_freq.keys()), list(topics_freq.values()), edgecolor='black')
    plt.xlabel('Topics')
    plt.ylabel('Frequency')
    plt.xticks(range(0, 90), rotation=90)
    plt.xticks(fontsize=8)
    plt.show()


def get_sorted_topics(categories, documents_topics):
    topics_freq = {topic: 0 for topic in categories}
    for topic_list in documents_topics:
        for topic in topic_list:
            topics_freq[categories[topic]] += 1
    topics_freq = dict(sorted(topics_freq.items(), key=lambda x: x[1], reverse=True))
    return topics_freq


def get_tf_from_corpus(document, term_id):
    for index in range(len(document)):
        if document[index][0] == term_id:
            return document[index][1]
        else:
            return 0


def get_number_words_in_topic(topic_term_probability):
    previous_probability = 0
    for index, pair in enumerate(topic_term_probability):
        if pair[1] != previous_probability:
            previous_probability = pair[1]
        else:
            break
    # Note: The element of the document we are interested about has index = index-1 but since we are interested in the
    # number(quantity) of the elements in the document, we return the index value which is equal to the count.
    return index


def calculate_coverage(num_documents, num_topics, num_words, model, corpus):
    n_tk = np.zeros(shape=num_topics)
    proportions = np.zeros(shape=num_topics)
    coverages = np.zeros(shape=num_documents)

    for topic in range(num_topics):
        n_tk[topic] = get_number_words_in_topic(model.get_topic_terms(topic, num_words))

    total_tf = [sum(freq for _, freq in corpus[doc]) for doc in range(num_documents)]

    for doc in range(num_documents):
        doc_topics, word_topics, _ = model.get_document_topics(bow=corpus[doc], per_word_topics=True)
        doc_topics = [topic_id for topic_id, _ in doc_topics]
        for topic in doc_topics:
            sum_words_topic_tf = 0
            document_topic_num_words = 0
            for word_id, word_topic_list in word_topics:
                if topic in word_topic_list:
                    document_topic_num_words += 1 # number of terms with specific topic in each document
                    sum_words_topic_tf += [item[1] for item in corpus[doc] if item[0] == word_id][0]
            proportions[topic] = sum_words_topic_tf / (n_tk[topic] - document_topic_num_words + 1)

        under_root = 0
        for word_id, word_tf in corpus[doc]:
            p1 = word_tf / total_tf[doc]
            p2 = 0
            for topic in doc_topics:
                p2 += (word_tf / sum_words_topic_tf) * proportions[topic] if sum_words_topic_tf != 0 else 0
            under_root += np.power(p1 - p2, 2)
        coverages[doc] = np.power(under_root, 0.5)
    return coverages


def calculate_preplexity(topic_word_probs, doc_topic_probs, tokenized_documents):
    # Calculate the log-likelihood of each document
    log_likelihoods = []
    document_lengths = []
    for doc_index, doc in enumerate(tokenized_documents):
        log_likelihood = 0
        for word_index in range(len(doc)):
            word_topic_probs = topic_word_probs[:, word_index]
            doc_topic_prob = [topic_prob for _, topic_prob in doc_topic_probs[doc_index]]
            word_likelihood = np.dot(word_topic_probs, doc_topic_prob)
            log_likelihood += np.log(word_likelihood)
        log_likelihoods.append(log_likelihood)
        document_lengths.append(len(doc))

    # Calculate the perplexity
    perplexity = (-1) * sum(log_likelihoods) / (10 * sum(document_lengths))

    # Calculate the perplexity for each document
    # perplexities = []
    # for log_likelihood, doc in zip(log_likelihoods, documents):
    #     perplexity = -log_likelihood / (10 * len(doc))
    #     perplexities.append(perplexity)

    # # Calculate the average perplexity across all documents
    # avg_perplexity = np.mean(perplexities)

    return perplexity


def tokenize(text):
    min_length = 0
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = (word for word in words if word not in cachedStopWords)
#    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    tokens = (PorterStemmer().stem(token) for token in words)
    p = re.compile('[a-zA-Z]+')
#    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    filtered_tokens = [token for token in tokens if p.match(token) and len(token) >= min_length]
    return filtered_tokens


def preprocess_manual(train_docs):

    tokenized_documents = {}

    for doc_id, document in train_docs.items():
        tokenized_doc = tokenize(document)
        if len(tokenized_doc) != 0:
            tokenized_documents[doc_id] = tokenized_doc

    unique_tokens = list(set(term for doc in tokenized_documents.values() for term in doc))

    return tokenized_documents, unique_tokens



def main():

    # un-comment to get all the documents
    doc_fileids = reuters.fileids()

    # retrieving specific topics
    categories = reuters.categories()
    topics = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']
    topics_indexes = [categories.index(topic) for topic in topics]
    doc_fileids = [fileid for topic in topics for fileid in reuters.fileids(topic)]

    # raw data and categories
    train_docs = list(filter(lambda doc: doc.startswith("train"), doc_fileids))
    # train_docs = train_docs
    test_docs = list(filter(lambda doc: doc.startswith("test"), doc_fileids))

    # delete out of study topic indexes
    for doc_topic in documents_topics:
        for topic in doc_topic:
            if topic not in topics_indexes:
                doc_topic.remove(topic)

    documents_topics, tokenized_documents, unique_tokens = Preprocess.preprocess_routers(categories, train_docs, minimum_doc_length=0)

    num_topics = len(topics)
    num_documents = len(tokenized_documents)
    num_words = len(unique_tokens)

    # Create a dictionary of unique words
    dictionary = corpora.Dictionary(tokenized_documents)

    # Convert the corpus into a bag-of-words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_documents]

    sorted_topics = get_sorted_topics(favorite_topics, documents_topics_ids_train)


    # Define the coherence metric
    # lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # lda.get_topic_terms(1, topn=num_words)
    # lda.get_document_topics(bow=corpus[0], minimum_probability=None, per_word_topics=True)
    # lda.get_topics()
    #

    # coverages = calculate_coverage(num_documents, num_topics, num_words, lda, corpus)
    # coverage_model = np.power(sum(np.power(coverages, 2)) / num_documents, 0.5)

    # Define the coherence metric
    # coherence_model = models.CoherenceModel(model=lda, texts=tokenized_documents, dictionary=dictionary, coherence='c_uci')
    # preplexity_model = lda.log_perplexity(corpus)

    # Define the ACO parameters
    num_ants = 5
    num_iterations = 40
    alpha = 0.4  # percentage taken from randomness
    beta = 0.6  # percentage taken from heuristic (pheromone)
    theta = 0.5
    evaporation_rate = 0.1

    # Define the pheromone matrix size=(num_words, num_topics)
    pheromone_matrix = np.ones(shape=(num_topics, len(dictionary)))
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=1, norm='l1')

    # Initialize the ant population
    ants = []
    for i in range(num_ants):
        ant = [uniform(0, 1) for _ in range(num_words)]  # probability of each topic
        ants.append(ant)

    # Run the ACO algorithm
    best_solution = None
    best_solution_score = float('inf')

    for iteration in range(num_iterations):
        for ant_index in range(num_ants):
            ant = []
            for i in range(num_topics):
                word = [uniform(0, 1) for _ in range(num_words)]  # probability of each topic
                ant.append(word)
            ant = np.array(ant)
            ant = preprocessing.normalize(ant, axis=1, norm='l1')

            # Generate a new topic distribution for the documents
            topic_distribution = np.multiply(ant * alpha, pheromone_matrix * beta)
            topic_distribution = preprocessing.normalize(topic_distribution, axis=1, norm='l1')
            # another formula is
            # topic_probability = (pheromone_value ** alpha) * ((term_probability + beta) ** beta)

            # Evaluate the quality of the new topic distribution
            new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto',
                                      eta=topic_distribution)

            coverages = calculate_coverage(num_documents, num_topics, num_words, new_lda, corpus)
            coverage_score = np.power(sum(np.power(coverages, 2)) / num_documents, 0.5)
            coherence_score = models.CoherenceModel(model=new_lda, texts=tokenized_documents_train, dictionary=dictionary,
                                                    coherence='c_uci').get_coherence() * -1

            # Get the topic-word probabilities matrix
            topic_word_probs = new_lda.get_topics()
            # Get the document-topic probabilities matrix
            doc_topic_probs = new_lda.get_document_topics(corpus, minimum_probability=0)
            preplexity_score = calculate_preplexity(topic_word_probs, doc_topic_probs, tokenized_documents_train)

            objective_values = (coverage_score, coherence_score, preplexity_score)
            theoretical_best_solution = (0, 0, 0)
            objective_value = np.sqrt(np.sum(np.square(np.subtract(objective_values, theoretical_best_solution))))

            # Update the best solution found so far
            if objective_value < best_solution_score:
                best_solution = topic_distribution
                best_solution_score = objective_value

            # Update the pheromone matrix
            for topic_id in range(num_topics):
                for word_id in range(len(dictionary)):
                    pheromone_matrix[topic_id, word_id] *= (1 - evaporation_rate)
                    term_topics = new_lda.get_term_topics(word_id)
                    if len(term_topics) > 0:
                        predicted_topic, probability = term_topics[0]
                        pheromone_matrix[predicted_topic, word_id] += probability
            pheromone_matrix = pheromone_matrix / preprocessing.normalize(pheromone_matrix, axis=1)

            print("ant: ", ant_index, " coherence", coherence_score, " coverage: ", coverage_score, " preplexity: ",
                  preplexity_score)

        # Update the pheromone matrix
        pheromone_matrix = np.add(pheromone_matrix, topic_distribution)
        pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=1, norm='l1')

        # Print the best solution found so far
        print('Iteration', iteration, 'Best objective function:', best_solution_score, 'Best solution:', best_solution)


np.savetxt("topic_word_probs.csv", topic_word_probs, delimiter=",")