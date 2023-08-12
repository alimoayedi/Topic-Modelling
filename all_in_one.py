from numba import jit, cuda
import numpy as np
import nltk
import tokenize
import re
import random
import matplotlib.pyplot as plt

from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('reuters')
# defined stop words
cachedStopWords = stopwords.words("english")

# raw data and categories
documents = reuters.fileids()
train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
categories = reuters.categories()
train_docs = train_docs[0:50]
categories = categories

# number of dimensions
num_topics = len(categories)
num_documents = len(train_docs)
num_words = 0

# calculated measurements
prop = np.array([])
coverage = 0
coherence = 0

# parameters
num_ants = 40
num_iterations = 5
alpha = 0.8
beta = 0.2
rho = 0.1
Q = 2

# matrices
document_topic_matrix = np.array([])
word_topic_matrix = np.array([])  # TODO, will be added again as self, to avoid matrix from returning it.
topic_probabilities = np.full(num_topics, 1/num_topics)
pheromone_matrix = np.array([])  # size = (number_words, number_topics)

best_coverages = []
best_solutions = []
global_best_coverage = []
global_best_solution = []

# doc_topic_plot
# doc_topic_fig, doc_topic_fig_axis = plt.subplots()
# plt.show()

# coverage plot
plt.ion()
coverage_figure, coverage_axis = plt.subplots()
coverage_scatter = coverage_axis.scatter([], [])
plt.show()



def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def preprocess(train_docs):
    tokenized_documents = []
    documents_topics = []
    for doc in train_docs:
        # tokenize the document
        tokenized_doc = tokenize(reuters.raw(doc))
        if len(tokenized_doc) != 0:
            tokenized_documents.append(tokenized_doc)
        documents_topics.append([categories.index(topic) for topic in reuters.categories(doc)])
    unique_tokens = list(set(term for doc in tokenized_documents for term in doc))
    return documents_topics, tokenized_documents, unique_tokens


def initialize_word_topic_matrix(bow_documents):  # initialize word_topic_matrix
    """
    INITIAL UNIFORM DISTRIBUTION
    Normalize the count of a given word assigned to a particular topic across all documents in the corpus.
    without normalization, longer documents would have a greater influence on the word-topic matrix,
    since they contain more words and therefore more counts of each word
    """
    word_topic_matrix = np.zeros(shape=(num_documents, num_words))
    for doc_index in range(num_documents):
        word_topic_matrix[doc_index] = np.random.choice(num_topics, size=num_words)
    return word_topic_matrix

    #
    # for doc in bow_documents:
    #     for word_index, word_count in enumerate(doc):
    #         for topic_index in range(num_topics):
    #             word_topic_matrix[word_index, topic_index] += word_count / len(bow_documents)
    #
    # # normalize word_topic_matrix
    # word_topic_matrix /= np.sum(word_topic_matrix, axis=0)
    # return word_topic_matrix


# @jit(target_backend='cuda')
def calculate_topic_probabilities(word_index):
    """
    Calculates the probability of each topic being assigned to a given word.

    Parameters:

    word_index (int): the index of the word in the vocabulary

    Returns:
    topic_probabilities (numpy.ndarray): a 1D array where each element represents the probability of a topic being assigned to the word
    """
    for index in range(num_topics):
        topic_probabilities[index] = word_topic_matrix[word_index, index] / np.sum(word_topic_matrix[:, index])
    return topic_probabilities


# @jit(target_backend='cuda')
def generate_ant_solution():
    ant_solution = [-1] * num_words
    for word_index in range(num_words):
        topic_probabilities = calculate_topic_probabilities(word_index)
        pheromone_probabilities = pheromone_matrix[word_index] ** alpha
        probabilities = (1 - beta) * pheromone_probabilities + beta * topic_probabilities
        probabilities /= sum(probabilities)
        ant_solution[word_index] = np.random.choice(np.arange(num_topics), p=probabilities)
    return ant_solution


def calculate_proportion(topic_term_document, ant_solutions):
    '''
    :param topic_term_document: the size is (topic, term, document)
    :param ant_solutions: the size is (document, term)
    :return:
    '''
    prop = np.zeros(shape=(num_topics, num_documents))
    # Number of terms in topic k
    terms_documents_matrix = np.array(
        [sum(np.array(document)).tolist() for document in zip(*topic_term_document)]).T.tolist()
    # calculates topics frequency through all documents
    n_tk = np.zeros(num_topics)

    for ant_solution in ant_solutions:
        for topic_index in range(num_topics):
            n_tk[topic_index] += ant_solution.count(topic_index)
    for doc_index in range(num_documents):
        document = terms_documents_matrix[doc_index]  # takes the document
        ant_solution_topics_df = np.zeros(shape=num_topics)  # how many times each topic is repeated in this document

        for pair in zip(document, ant_solution):
            if pair[0]:  # pair[0]=term count
                ant_solution_topics_df[pair[1]] += pair[0]  # pair[1]=predicted topic by ant. Increase topic count
        for topic_index in range(num_topics):
            count = ant_solution.count(topic_index)
            prop[topic_index, doc_index] = ant_solution_topics_df[topic_index] / (n_tk[topic_index] - count + 1)
    return prop


# @jit(target_backend='cuda')
def calculate_coverage(topic_term_document, ant_solutions, prop):
    # get prop from the constructor
    documents_coverage = np.zeros(shape=(num_documents))
    for doc in range(num_documents):
        under_root = 0  # from the formula, it is the total before getting root!
        number_each_term_in_document = sum(topic_term_document[:, :, doc])  # number of each term in the document
        total_num_terms_in_document = sum(sum(topic_term_document[:, :, doc]))  # total number of terms in the doc
        weights = number_each_term_in_document / total_num_terms_in_document
        weights_topics_pair = list(zip(weights, ant_solutions[doc]))
        for term in range(num_words):
            p2 = 0  # second part of the formula(summation of multiplication of weights by prop)
            weight, ant_topic = weights_topics_pair[term]
            # weight * prop[topic, doc]
            # weight, topic = weights_topics_pair[term]
            for topic in range(num_topics):
                if topic == ant_topic:
                    p2 += weight * prop[topic, doc]
            p1 = number_each_term_in_document[term] / total_num_terms_in_document
            under_root += (p1 - p2) ** 2
        documents_coverage[doc] = under_root ** 0.5
    return documents_coverage
    # np.sqrt(sum(np.power(documents_coverage, 2)) / num_documents)


# def run(train_docs):
#     # initialization
#     documents_topics, tokenized_documents, unique_tokens = preprocess(train_docs)
#     num_documents = len(tokenized_documents)
#     num_words = len(unique_tokens)
#     document_topic_matrix = np.zeros((num_documents, num_topics))
#     topic_term_document = np.zeros((num_topics, num_words, num_documents), dtype=int)
#     pheromone_matrix = np.ones(shape=(num_words, num_topics)) / num_topics

#     for doc_index, doc in enumerate(tokenized_documents):
#         for term in doc:
#             term_index = unique_tokens.index(term)
#             for topic_index in documents_topics[doc_index]:
#                 topic_term_document[topic_index, term_index, doc_index] += 1

#     # bow_document is the number of term in each document over all topics (sum over all topics)
#     bow_documents = [[sum(topics) for topics in zip(*topic_document_set)] for topic_document_set in
#                      zip(*(topic_term_document.tolist()))]
#     # transpose bow_documents
#     bow_documents = np.array(bow_documents).T.tolist()
#     # initialize word_topic_matrix
#     word_topic_matrix = initialize_word_topic_matrix(topic_term_document, bow_documents)

#     for iteration in range(num_iterations):
#         for _ in range(num_ants):
#             # assign a random topic to each of the documents, size = (term_count, doc_count)
#             ant_solutions = [generate_ant_solution() for _ in range(num_documents)] # TODO should I remove bow_doc from here? check generate function
#             proportion = calculate_proportion(topic_term_document, ant_solutions)
#             documents_coverage = calculate_coverage(topic_term_document, ant_solutions, proportion)
#             objective_values = documents_coverage
#             local_minimum_coverage_index = np.argmin(objective_values)
#             local_best_solution = ant_solutions[local_minimum_coverage_index]
#             best_coverages.extend([objective_values[local_minimum_coverage_index]])
#             best_solutions.append(local_best_solution)
#             pheromone_matrix = update_pheromone_matrix(local_best_solution, min(objective_values), pheromone_matrix)

#         global_best_coverage = min(best_coverages)
#         global_best_solution = best_solutions[np.argmin(global_best_coverage)]
#         update_document_topic_matrix(global_best_solution, bow_documents)
#         update_word_topic_matrix(global_best_solution, unique_tokens)
#         to_string(document_topic_matrix)
#         print('wait...')


def update_pheromone_matrix(ant_solution, coverage, pheromone_matrix):
    delta_pheromone_matrix = np.zeros(shape=(num_words, num_topics))
    for i in range(len(ant_solution)):  # len(ant_solution) == num_words
        topic_index = ant_solution[i]
        delta_pheromone_matrix[i, topic_index] += coverage
    pheromone_matrix = (1 - rho) * pheromone_matrix + Q * delta_pheromone_matrix
    return pheromone_matrix


def update_document_topic_matrix(best_ant_solution, bow_documents):
    for doc_index in range(num_documents):
        total_term_count_in_doc = sum(bow_documents[doc_index])
        doc_nonzero_terms_indexes = np.nonzero(bow_documents[doc_index])[0]
        for term_index in doc_nonzero_terms_indexes:
            predicted_topic_index = best_ant_solution[term_index]
            term_count = bow_documents[doc_index][term_index]
            document_topic_matrix[doc_index][predicted_topic_index] = term_count / total_term_count_in_doc


def update_word_topic_matrix(best_ant_solution, bow_documents):
    for doc_index in range(num_documents):
        nonzero_terms_indexes = np.nonzero(bow_documents[doc_index])[0]
        for term_index in nonzero_terms_indexes:
            topic_index = best_ant_solution[term_index]
            word_topic_matrix[term_index][topic_index] += 1


def to_string(matrix):
    for row in matrix:
        print('\t'.join(map(str, np.round(row, 2))))


# @jit(target_backend='cuda')
# def plot_doc_topic_matrix(document_topic_matrix):
#     # colors = [[(1, 1-val, 1-val) for val in doc] for doc in document_topic_matrix]
#     doc_topic_fig_axis.scatter(x=[i for i in range(num_documents) for _ in range(num_topics)],
#                                y=[i for _ in range(num_documents) for i in range(num_topics)],
#                                s=[topic_prob * 1000 for doc in document_topic_matrix for topic_prob in doc])
#     # multiply by 1000 to increase circle size)
#     plt.show()



# @jit(target_backend='cuda')
def plot_coverage(best_coverages):
    x_iteration = [i for i in range(len(best_coverages))]
    coverage_scatter.set_offsets(np.c_[x_iteration, best_coverages])
    coverage_axis.set_ylim([0, max(best_coverages)])
    coverage_axis.set_xlim([0, max(x_iteration)])
    coverage_figure.canvas.draw_idle()
    plt.pause(0.1)



def run(train_docs):
    # initialization
    documents_topics, tokenized_documents, unique_tokens = preprocess(train_docs)
    num_documents = len(tokenized_documents)
    num_words = len(unique_tokens)
    document_topic_matrix = np.zeros((num_documents, num_topics))
    topic_term_document = np.zeros((num_topics, num_words, num_documents), dtype=int)
    pheromone_matrix = np.ones(shape=(num_words, num_topics)) / num_topics

    for doc_index, doc in enumerate(tokenized_documents):
        for term in doc:
            term_index = unique_tokens.index(term)
            for topic_index in documents_topics[doc_index]:
                topic_term_document[topic_index, term_index, doc_index] += 1

    # bow_document is the number of term in each document over all topics (sum over all topics)
    bow_documents = [[sum(topics) for topics in zip(*topic_document_set)] for topic_document_set in
                     zip(*(topic_term_document.tolist()))]
    # transpose bow_documents
    bow_documents = np.array(bow_documents).T.tolist()
    # initialize word_topic_matrix
    word_topic_matrix = initialize_word_topic_matrix(bow_documents)

    # plot_doc_topic_matrix(document_topic_matrix)

    for iteration in range(num_iterations):
        for ant_index in range(num_ants):
            print(i)
            # assign a random topic to each of the documents, size = (term_count, doc_count)
            calculate_topic_probabilities()  # TODO how to update?
            ant_solutions = [generate_ant_solution() for _ in range(num_documents)]  # TODO should I remove bow_doc from here? check generate function
            proportion = calculate_proportion(topic_term_document, ant_solutions)
            documents_coverage = calculate_coverage(topic_term_document, ant_solutions, proportion)
            objective_values = documents_coverage  # TODO other functions come here
            local_minimum_coverage_index = np.argmin(objective_values)
            local_best_solution = ant_solutions[local_minimum_coverage_index]
            best_coverages.extend([objective_values[local_minimum_coverage_index]])  # TODO should this variable gets empty at each iteration?
            best_solutions.append(local_best_solution)
            plot_coverage(best_coverages)
            pheromone_matrix = update_pheromone_matrix(local_best_solution, min(objective_values), pheromone_matrix)

        # TODO check if the new solution is better than the global one or not?!
        global_best_coverage = min(best_coverages)
        global_best_solution = best_solutions[np.argmin(global_best_coverage)]
        update_document_topic_matrix(global_best_solution, bow_documents)
        update_word_topic_matrix(global_best_solution, unique_tokens)
        # plot_doc_topic_matrix(document_topic_matrix)
        # to_string(document_topic_matrix)
        print('wait...')
        ant_solutions = [generate_ant_solution(bow_document) for bow_document in bow_documents]

        to_string(word_topic_matrix)

        np.savetxt('document_topic_matrix.csv', document_topic_matrix, delimiter=',', fmt='%f')
