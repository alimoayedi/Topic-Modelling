import numpy as np
import random
import re
import os
import nltk
import matplotlib.pyplot as plt
import statistics as st

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import pdb


nltk.download('punkt')
nltk.download('stopwords')

# defined stop words
cachedStopWords = stopwords.words("english")



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
    plt.xticks(range(min(sublist_lengths), max(sublist_lengths) + 1,
                     (max(sublist_lengths) - min(sublist_lengths) + 1) // num_ticks))
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
    plt.xticks(range(0, len(categories)), rotation=90)
    plt.xticks(fontsize=8)
    plt.show()


def get_sample(data_list, percentage):
    # Calculate the total length of arrays tokenized_documents
    total_length = len(data_list)

    # Determine the number of items to select based on percentage
    num_documents_to_select = int(total_length * percentage)

    # Shuffle the indices randomly
    random.shuffle(data_list)

    # Select the first num_items_to_select indices
    sample_data = data_list[:num_documents_to_select]

    return sample_data


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

    # generate a list of all terms (from each terms ONE sample should be kept)
    unique_tokens = list(set(term for doc in tokenized_documents.values() for term in doc))

    return tokenized_documents, unique_tokens


def get_train_test_data(data_list, percentage):
    # Calculate the total length of arrays tokenized_documents
    total_length = len(data_list)

    # Determine the number of items to select (80% of total length)
    num_documents_to_select = int(total_length * percentage)

    # Shuffle the indices randomly
    random.shuffle(data_list)

    # Select the first num_items_to_select indices
    train_data = data_list[:num_documents_to_select]
    test_data = data_list[num_documents_to_select:]

    return train_data, test_data


def get_topics_distribution(categories, documents_topics):
    topics_freq = {topic: 0 for topic in categories}
    for topic_list in documents_topics:
        for topic in topic_list:
            topics_freq[categories[topic]] += 1
    topics_freq = dict(sorted(topics_freq.items(), key=lambda x: x[1], reverse=True))
    return topics_freq


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


def get_number_words_in_topic(topic_term_probability):
    count = sum(1 for pair in topic_term_probability if pair[1] > 9e-5)
    return count


def calculate_coverage(num_documents, num_topics, num_words, model, corpus):
    n_tk = np.zeros(shape=num_topics)
    proportions = np.zeros(shape=num_topics)
    coverages = np.zeros(shape=num_documents)

    # finds the number of terms each topic has.
    for topic in range(num_topics):
        n_tk[topic] = get_number_words_in_topic(model.get_topic_terms(topic, num_words))

    # finds the total number of terms in each document
    total_tf = [sum(freq for _, freq in corpus[doc]) for doc in range(num_documents)]

    for doc in range(num_documents):
        doc_topics, word_topics, _ = model.get_document_topics(bow=corpus[doc], per_word_topics=True)
        doc_topics = [topic_id for topic_id, _ in doc_topics]
        for topic in doc_topics:
            sum_words_topic_tf = 0
            document_topic_num_words = 0
            for word_id, word_topic_list in word_topics:
                if topic in word_topic_list:
                    document_topic_num_words += 1  # number of terms with specific topic in each document
                    sum_words_topic_tf += [item[1] for item in corpus[doc] if item[0] == word_id][0]
            if document_topic_num_words > n_tk[topic]:
                n_tk[topic] = document_topic_num_words
                print("**** An error in number of topic terms occurred. ****")
            proportions[topic] = sum_words_topic_tf / ((n_tk[topic] - document_topic_num_words) + 1)


        under_root = 0
        for word_id, word_tf in corpus[doc]:
            p1 = word_tf / total_tf[doc]
            p2 = 0
            for topic in doc_topics:
                p2 += (word_tf / sum_words_topic_tf) * proportions[topic] if sum_words_topic_tf != 0 else 0
            under_root += np.power(p1 - p2, 2)
        coverages[doc] = np.power(under_root, 0.5)
    return coverages


def roulette_wheel(ants_fitness):
    num_ants = len(ants_fitness)
    fitness_ranks = np.unique(ants_fitness, return_inverse=True)[1] + 1
    scaled_fitness = np.divide(ants_fitness, fitness_ranks)
    sum_scaled_fitness = np.sum(scaled_fitness)
    selection_probs = np.divide(scaled_fitness, sum_scaled_fitness)
    return np.random.choice(num_ants, p=selection_probs, size=num_ants)

def file_exists(filename):
    """Check if a file exists in the current directory"""
    directory = f'C:/Thesis/saved_status/{filename}.npy'
    return os.path.exists(directory) and os.path.isfile(directory)

