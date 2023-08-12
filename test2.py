import numpy as np
import nltk
import tokenize
import re
import random

from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('reuters')
# defined stop words
cachedStopWords = stopwords.words("english")

documents = reuters.fileids()
train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
categories = reuters.categories()


class AntColony:
    def __init__(train_docs, categories, num_ants, num_iterations, alpha, beta, rho, Q):
        train_docs = train_docs[0:50]
        categories = categories
        num_topics = len(categories)
        num_documents = len(train_docs)
        num_words = 0
        prop = np.array([])
        coverage = 0
        coherence = 0

        num_ants = 10
        num_iterations = 10
        alpha = 0.5
        beta = 0.5
        rho = 0.5
        Q = 1
        pheromone_matrix = np.array([])
        document_topic_matrix = np.array([])
        word_topic_matrix = np.array([])  #TODO, will be added again as self, to avoid matrix from returning it.
        topic_probabilities = np.zeros(num_topics)

    @staticmethod
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
    
    def run(train_docs):
        documents_topics, tokenized_documents, unique_tokens = preprocess(train_docs)
        num_documents = len(tokenized_documents)
        num_words = len(unique_tokens)
        
        pheromone_matrix = np.ones(shape=(num_words, num_topics)) / num_topics
        document_topic_matrix = np.zeros((num_documents, num_topics))
        document_term_topic = np.zeros((num_topics, num_words, num_documents), dtype=int)

        for index, doc in enumerate(tokenized_documents):
            for term in doc:
                term_index = unique_tokens.index(term)
                for topic_index in documents_topics[0]:
                    document_term_topic[topic_index, term_index, index] += 1

        # bow_document is the number of term in each document over all topics (sum over all topics)
        bow_documents = [[sum(topics) for topics in zip(*topic_document_set)] for topic_document_set in zip(*(document_term_topic.tolist()))]
        # transpose bow_documents
        bow_documents = np.array(bow_documents).T.tolist()
        # initialize word_topic_matrix
        word_topic_matrix = initialize_word_topic_matrix(bow_documents)
        # assign a random topic to each of the documents, size = (term_count, doc_count)
        ant_solutions = [initialize_ant_solution(bow_document) for bow_document in bow_documents]


        for iteration in range(num_iterations):
            documents_coverage = calculate_coverage(ant_solutions)
            objective_values = documents_coverage
            for idx, ant_solution in enumerate(ant_solutions):
                update_pheromone_matrix(ant_solution, documents_coverage[idx])

            best_ant_solution = ant_solutions[np.argmax(objective_values)]
            update_document_topic_matrix(best_ant_solution, bow_documents)
            update_word_topic_matrix(best_ant_solution, unique_tokens)
            calculate_topic_probabilities()
            ant_solutions = [construct_ant_solution(bow_document) for bow_document in bow_documents]

        return document_topic_matrix, word_topic_matrix

    def initialize_word_topic_matrix(bow_documents):  # initialize word_topic_matrix
        """
        INITIAL UNIFORM DISTRIBUTION
        Normalize the count of a given word assigned to a particular topic across all documents in the corpus.
        without normalization, longer documents would have a greater influence on the word-topic matrix,
        since they contain more words and therefore more counts of each word
        """
        word_topic_matrix = np.zeros(shape=(num_words, num_topics))
        for doc in bow_documents:
            for j, word_count in enumerate(doc):
                for k in range(num_topics):
                    word_topic_matrix[j, k] += word_count / len(bow_documents)

        # normalize word_topic_matrix
        word_topic_matrix /= np.sum(word_topic_matrix, axis=0)
        return word_topic_matrix

    def initialize_ant_solution(bow_document):
        ant_solution = [-1] * len(bow_document)
        for word_index in range(len(bow_document)):
            topic_probabilities = calculate_topic_probabilities(word_index)
            pheromone_probabilities = pheromone_matrix[word_index] ** alpha
            probabilities = (1 - beta) * pheromone_probabilities + beta * topic_probabilities
            probabilities /= sum(probabilities)
            ant_solution[word_index] = np.random.choice(np.arange(num_topics), p=probabilities)
        return ant_solution


    def calculate_topic_probabilities(word_index):
        """
        Calculates the probability of each topic being assigned to a given word.

        Parameters:
            
        word_index (int): the index of the word in the vocabulary

        Returns:
        topic_probabilities (numpy.ndarray): a 1D array where each element represents the probability of a topic being assigned to the word
        """
        # TODO needs word_topic_matrix
        topic_probabilities = np.zeros(num_topics)
        for j in range(num_topics):
            topic_probabilities[j] = word_topic_matrix[word_index, j] / np.sum(word_topic_matrix[:, j])
        return topic_probabilities

    # def convert_to_bow(document, vocabulary):
    #     bow = [0] * num_words
    #     for word in document:
    #         if word in vocabulary:
    #             bow[vocabulary.index(word)] += 1
    #     return bow

    def calculate_proportion(document_term_topic):
        prop = np.zeros(shape=(num_topics, num_documents))
        for topic in range(num_topics):
            # Number of terms in topic k
            n_tk = np.count_nonzero(document_term_topic[topic, :, :])
            for document in range(num_documents):
                # Summation of the tf of terms in document under the topic (numerator)
                sum_tf_term = sum(document_term_topic[topic, :, document])
                # number of terms in the document under the topic
                num_term_in_doc = np.count_nonzero(document_term_topic[topic, :, document])
                prop[topic][document] += (sum_tf_term / (n_tk - num_term_in_doc + 1))

    def calculate_coverage(document_term_topic):
        # get prop from the constructor
        documents_coverage = np.zeros(shape=(num_documents))
        for doc in range(num_documents):
            under_root = 0  # from the formula, it is the total before getting root!
            p2 = 0  # second part of the formula(summation of multiplication of weights by prop)
            count_each_term_in_document = sum(document_term_topic[:, :, doc])  # count of each term in the document
            total_num_terms_in_document = sum(sum(document_term_topic[:, :, doc]))  # total number of terms in the doc
            for term in range(count_each_term_in_document):
                for topic in range(num_topic):
                    weight = document_term_topic[topic, term, doc] / document_term_topic[topic, :, doc]
                    p2 += weight * prop[topic, doc]
                p1 = count_each_term_in_document[term] / total_num_terms_in_document
                under_root += (p1 - p2)**2
            documents_coverage[doc] = under_root ** 0.5
        return documents_coverage
        # np.sqrt(sum(np.power(documents_coverage, 2)) / num_documents)

    def calculate_coherence(document_term_topic):
        for topic in range(num_topics):
            pmi = 0
            terms_in_topic = np.sum(document_term_topic[topic, :, :], 1)
            total_num_term = sum(terms_in_topic)
            combinations = [(a, b) for idx, a in enumerate(terms_in_topic) for b in terms_in_topic[idx + 1:]]
            num_relevant_terms_co_occurrences = 0
            for a,b in combinations: # (a,b) a = first term's count under topic, b = second term's count
                if a and b: # a and b not zero
                    p_a = a / total_num_term
                    p_b = b / total_num_term
                    p_ab = min(a, b) / total_num_term
                    co_occurrence.append(np.log10(p_a) + np.log10(p_b)) / np.log10(p_ab)
                    num_relevant_terms_co_occurrences += 1
                else:
                    co_occurrence = -1
                pmi += co_occurrence
            pmi = pmi / num_relevant_terms_co_occurrences
            coherence = coherence + pow(1-pmi, 2)
        coherence = np.sqrt(coherence)

    def update_pheromone_matrix(ant_solution, coverage):
        delta_pheromone_matrix = np.zeros((num_topics, num_words))
        for i in range(len(ant_solution)):
            for j in range(len(ant_solution[i])):
                delta_pheromone_matrix[ant_solution[i][j], j] += coverage
        pheromone_matrix = (1 - rho) * pheromone_matrix + Q * delta_pheromone_matrix

    def update_document_topic_matrix(ant_solution, bow_documents):
        for i in range(len(ant_solution)):
            for j in range(len(ant_solution[i])):
                document_topic_matrix[i, ant_solution[i][j]] += bow_documents[i][j]

    def update_word_topic_matrix(ant_solution, vocabulary):
        for i in range(len(ant_solution)):
            for j in range(len(ant_solution[i])):
                word_topic_matrix[vocabulary.index(words[i][j]), ant_solution[i][j]] += 1

    def construct_ant_solution(bow_document):
        ant_solution = []
        for i in range(len(bow_document)):
            probabilities = calculate_probabilities(bow_document[i])
            topic = select_topic
