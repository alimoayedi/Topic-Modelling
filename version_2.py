from nltk import word_tokenize
from numba import jit, cuda
import numpy as np
import nltk
import Tokenization
import matplotlib.pyplot as plt

from nltk.corpus import reuters

nltk.download('reuters')


# doc_topic_plot
# doc_topic_fig, doc_topic_fig_axis = plt.subplots()
# plt.show()

# coverage plot
plt.ion()
coverage_figure, coverage_axis = plt.subplots()
coverage_scatter = coverage_axis.scatter([], [])
plt.show()


def preprocess(categories, train_docs):
    tokenized_documents = []
    documents_topics = []
    for doc in train_docs:
        # tokenize the document
        tokenized_doc = Tokenization.tokenize(reuters.raw(doc))
        if len(tokenized_doc) != 0:
            tokenized_documents.append(tokenized_doc)
        documents_topics.append([categories.index(topic) for topic in reuters.categories(doc)])
    unique_tokens = list(set(term for doc in tokenized_documents for term in doc))
    return documents_topics, tokenized_documents, unique_tokens


def initialize_word_topic_probabilities(num_words, num_topics, word_topic_matrix):
    word_topic_probability = np.zeros(shape=(num_words, num_topics))
    for index, word_array in enumerate(word_topic_matrix):
        word_topic_probability[index,:] = list(map(lambda count: count/sum(word_array), word_array))
    return word_topic_probability


def get_probabilities(word_topic_probability, local_pheromone, global_pheromone, alpha, beta, rho):
    probabilities = (alpha * word_topic_probability) + (beta * local_pheromone) + rho * global_pheromone
    probabilities = [(word_probabilities / sum(word_probabilities)).tolist() for word_probabilities in probabilities]
    return probabilities


def generate_solution(num_words, num_topics, probabilities):
    solution = np.full(shape = num_words, fill_value=-1)  # -1 as the topic of each word (meaningless)
    for word_index in range(num_words):
        solution[word_index] = np.random.choice(np.arange(num_topics), size=None, p=probabilities[word_index])
    return solution


def calculate_proportion(num_topics, solutions, document_word_tf, n_tk):
    proportion = np.zeros(shape=(len(solutions), num_topics))
    for solution_index, solution in enumerate(solutions):
        solution_word_frequency_zip = list(zip(document_word_tf, solution))
        for topic in range(num_topics):
            tf_topic_pair = list(filter(lambda x: x[0] != 0 and x[1] == topic, solution_word_frequency_zip))
            tf_summation = 0
            for word in tf_topic_pair:
                tf_summation += word[0]
            proportion[solution_index, topic] = 0 if (n_tk[topic] == 0) else tf_summation / (n_tk[topic] - len(tf_topic_pair) + 1)
    return proportion


# @jit(target_backend='cuda')
def calculate_coverage(num_words, num_topics, solutions, proportions, word_topic_weight, word_document):
    coverages = np.zeros(shape=(len(solutions)))
    for solution_index, solution in enumerate(solutions):


        under_root = 0
        for word in range(num_words):
            p2 = 0  # second part of equation
            for topic in range(num_topics):
                weight = word_topic_weight[word, topic]
                p2 += weight * proportions[solution_index, topic]
            p1 = word_document[word] / np.sum(word_document)
            under_root += (p1 - p2) ** 2
        coverages[solution_index] = under_root ** 0.5
    return coverages


def get_joint_probabilities(num_topics, num_words, tokenized_documents, unique_tokens, documents_topics):
    topic_words_joint_probability = np.zeros(shape=(num_topics, num_words, num_words))
    for doc_index, doc in enumerate(tokenized_documents):
        for topic in documents_topics[doc_index]:
            pairwised_doc = [doc[word: word + 2] for word in range(len(doc) - 1)]
            for word in doc:
                word_index = unique_tokens.index(word)
                topic_words_joint_probability[topic, word_index, word_index] += 1
            for pair in pairwised_doc:
                word_index_1 = unique_tokens.index(pair[0])
                word_index_2 = unique_tokens.index(pair[1])
                topic_words_joint_probability[topic, word_index_1, word_index_2] += 1
                topic_words_joint_probability[topic, word_index_2, word_index_1] += 1

    corpus_tf = np.sum(topic_words_joint_probability, axis=0)
    total_num_terms = np.trace(corpus_tf)
    for topic in range(num_topics):
        topic_words_joint_probability[topic, :, :] = np.divide(topic_words_joint_probability[topic, :, :], total_num_terms)
    return topic_words_joint_probability


def calculate_coherence(num_topics, document_word_tf, solutions, joint_probabilities):
    coherences = np.zeros(shape=(len(solutions)))
    for solution_index, solution in enumerate(solutions):
        tf_topic_pairs = zip(document_word_tf, solution)
        solution_coherence = 0
        for topic in range(num_topics):
            pmi = 0
            topic_relevant_words = [index for index, pair in enumerate(tf_topic_pairs) if pair[1] == topic]
            if len(topic_relevant_words) > 0:
                combinations = [(word_1, word_2) for idx, word_1 in enumerate(topic_relevant_words) for word_2 in topic_relevant_words[idx + 1:]]
                co_occurrences = 0
                for term_combination in combinations:
                    p_a = joint_probabilities[topic, term_combination[0], term_combination[0]]
                    p_b = joint_probabilities[topic, term_combination[1], term_combination[1]]
                    p_ab = joint_probabilities[topic, term_combination[0], term_combination[1]]
                    if p_ab == 0:
                        co_occurrences += -1
                    else:
                        co_occurrences += ((np.log10(p_a) + np.log10(p_b)) / np.log10(p_ab))-1
                binomial_coefficient = (len(combinations) * (len(combinations)-1)) / 2
                pmi = co_occurrences / binomial_coefficient
            solution_coherence = solution_coherence + pow(1 - pmi, 2)
        coherences[solution_index] = np.sqrt(solution_coherence)
    return coherences


def update_local_pheromone(best_solution, local_pheromone, decay_rate):
    for term, topic in enumerate(best_solution):
        local_pheromone[term, topic] += 1
    return local_pheromone * (1 - decay_rate)


def update_global_pheromone(global_best_solution, global_pheromone):
    for term, topic in enumerate(global_best_solution):
        global_pheromone[term, topic] += 1
    return global_pheromone
    

def __str__(matrix):
    print('\t'.join(map(str, np.round(matrix, 2))))


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


def main():
    # raw data and categories
    documents = reuters.fileids()
    train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
    categories = reuters.categories()
    train_docs = train_docs[0:50]

    # number of dimensions
    documents_topics, tokenized_documents, unique_tokens = preprocess(categories, train_docs)
    num_topics = len(categories)
    num_documents = len(tokenized_documents)
    num_words = len(unique_tokens)

    # matrices
    topic_word_document = np.zeros((num_topics, num_words, num_documents), dtype=int)
    document_topic_matrix = np.zeros(shape=(num_documents, num_topics),dtype=int)
    word_topic_matrix = np.empty(shape=(num_words, num_topics), dtype=int)
    word_topic_probability = np.empty(shape=(num_words, num_topics))
    global_pheromone = np.ones(shape=(num_words, num_topics))
    local_pheromone = np.ones(shape=(num_words, num_topics))

    # parameters
    num_ants = 40
    num_iterations = 100
    alpha = 0.2  # effect of word_topic probabilities from the train data
    beta = 0.5  # effect of local pheromone comes from the best founded solution in each iteration
    rho = 0.3  # effect of global pheromone comes from the best solution of each document
    decay_rate = 0.2

    # calculated measurements
    proportion = np.array([])
    coverages = np.empty(shape=num_ants)
    coherences = np.empty(shape=num_ants)
    local_best_objective = 0
    local_best_solution = np.empty(shape=num_words)
    global_best_objective = 1000
    global_best_solution = np.empty(shape=num_words)
    global_best_coverages_plot = []

    # number of terms in each document under of each topic
    for doc_index, doc in enumerate(tokenized_documents):
        for term in doc:
            term_index = unique_tokens.index(term)
            for topic_index in documents_topics[doc_index]:
                topic_word_document[topic_index, term_index, doc_index] += 1

    # bow_document is the total number of term in each document over all topics (sum over all topics)
    # bow_documents = [[sum(topics) for topics in zip(*topic_document_set)] for topic_document_set in zip(*(topic_word_document.tolist()))]

    # transpose bow_documents from word_document to document_word
    # bow_documents = np.array(bow_documents).T.tolist()

    # calculates the number (count) of each word in each document
    word_document_matrix = np.array([sum(np.array(document)).tolist() for document in zip(*topic_word_document)])

    # number of each word under each topic over all the documents
    # (original result is topic_word. we transposed it)
    word_topic_matrix = np.array([[sum(word) for word in word_document] for word_document in topic_word_document]).T

    # word_topic probabilities
    word_topic_probability = initialize_word_topic_probabilities(num_words, num_topics, word_topic_matrix)

    # number of terms that the topic t_k has (!= term frequency)
    n_tk = np.count_nonzero(word_topic_matrix, axis=0)

    # joint_probability
    topic_words_joint_probability = get_joint_probabilities(num_topics, num_words, tokenized_documents, unique_tokens, documents_topics)

    for doc_index in range(num_documents):
        for i in range(num_iterations):
            probabilities = get_probabilities(word_topic_probability, local_pheromone, global_pheromone, alpha, beta, rho)
            solutions = [generate_solution(num_words, num_topics, probabilities) for _ in range(num_ants)]
            document_word_tf = word_document_matrix[:, doc_index]
            proportions = calculate_proportion(num_topics, solutions, document_word_tf, n_tk)
            a = topic_word_document[:, :, doc_index].T.astype(float)
            b = np.sum(topic_word_document[:, :, doc_index].T, axis=0).reshape(1, -1).astype(float)
            word_topic_weight = np.divide(a, b, out=a, where=b != 0)

            # word_topic_weight = topic_word_document[:, :, doc_index].T / np.sum(topic_word_document[:, :, doc_index].T, axis=0).reshape(1, -1)
            coverages = calculate_coverage(num_words, num_topics, solutions, proportions, word_topic_weight, word_document_matrix[:, doc_index])
            coherences = calculate_coherence(num_topics, document_word_tf, solutions, topic_words_joint_probability)

            coverage_coherence_pair = np.array(list(zip(coverages, coherences)))
            theoretical_best_solution = [(0, 0)] * num_ants
            summation_square = np.sum(np.square(coverage_coherence_pair - theoretical_best_solution), axis=1)
            objective_values = np.sqrt(summation_square)

            print(i)
            print(coverages)
            print(coherences)
            print(objective_values)
            # save best objective values
            best_solution_index = np.argmin(objective_values)
            local_best_objective = objective_values[best_solution_index]
            local_best_solution = solutions[best_solution_index]
            # update best global values
            if global_best_objective > local_best_objective:
                global_best_objective = local_best_objective
                global_best_solution = local_best_solution
            # update pheromones and weights
            local_pheromone = update_local_pheromone(local_best_solution, local_pheromone, decay_rate)
        global_best_coverages_plot.extend([global_best_objective])
        # plot_coverage(global_best_coverages_plot)
        __str__(global_best_coverages_plot)
        global_pheromone = update_global_pheromone(global_best_solution, global_pheromone)
