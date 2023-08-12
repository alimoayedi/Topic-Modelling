import numpy as np
import nltk
import tokenize
import re
import csv

from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('reuters')
# defined stop words
cachedStopWords = stopwords.words("english")


class AntColonyTopicModelling:
    def __init__(self, num_topics, num_words, num_documents, num_ants, alpha=1, beta=2, rho=0.1, Q=1):
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_documents = num_documents
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone = np.ones((num_topics, num_words)) / num_topics
        self.probability = np.zeros((num_topics, num_words, num_documents))
        self.documents = []
        self.words_in_document = []
        self.topics_in_document = []

    def tokenize(text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in cachedStopWords]
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
        return filtered_tokens

    def preprocessing(self):
        documents = reuters.fileids()
        train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
        categories = reuters.categories()




        # number of categories
        num_topics = len(reuters.categories())
        # cat_doc = np.zeros(shape=(num_topics, 1), dtype=int)
        # term_doc_dic = {}
        # term_cat_dic = {}
        # doc_count = 0

        # for doc in train_docs:
        #     # tokenize the document
        #     tokenized_doc = tokenize(reuters.raw(doc))
        #     if len(tokenized_doc) != 0:
        #         doc_count += 1
        #         cat_indexes = []
        #         # get all the categories and save them
        #         for cat in reuters.categories(doc):
        #             cat_indexes.append(categories.index(cat))
        #         # increase the freq. of categories in specific indexes
        #         for index in cat_indexes:
        #             cat_doc[index] += 1
        #         # if any term already exists in the dictionary one document is added to it.
        #         for key in term_doc_dic.keys():
        #             term_doc_dic[key] = np.append(term_doc_dic[key], 0)
        #         #
        #         for term in tokenized_doc:
        #             if term not in term_doc_dic:
        #                 term_doc_dic.update({term: np.zeros(doc_count, dtype=int)})  # add term to the document dictionary
        #                 term_cat_dic.update({term: np.zeros(num_topics, dtype=int)})  # add term to the category dictionary
        #             # increase term and category counts by 1
        #             term_doc_dic[term][-1] += 1
        #             for index in cat_indexes:
        #                 term_cat_dic[term][index] += 1
        #
        # term_doc_matrix = np.array([term_doc_dic[keyValue] for keyValue in term_doc_dic.keys()])
        # term_cat_matrix = np.array([term_cat_dic[keyValue] for keyValue in term_cat_dic.keys()])

        ##--------------------------------------------------------

        tokenized_documents = []
        documents_topics = []
        for doc in train_docs[0:100]:
            # tokenize the document
            tokenized_doc = tokenize(reuters.raw(doc))
            if len(tokenized_doc) != 0:
                tokenized_documents.append(tokenized_doc)
            documents_topics.append([categories.index(topic) for topic in reuters.categories(doc)])
        unique_tokens = list(set(term for doc in tokenized_documents for term in doc))
        no_of_documents = len(tokenized_documents)
        no_of_terms = len(unique_tokens)

        document_term_topic = np.zeros((len(categories), no_of_terms, no_of_documents), dtype=int)

        for index, doc in enumerate(tokenized_documents):
            for term in doc:
                term_index = unique_tokens.index(term)
                for topic_index in documents_topics[0]:
                    document_term_topic[topic_index, term_index, index] += 1

    document_term_topic[:,:,0]
    for topic in range()


        xs = [[[0, 0, 0], [1, 0, 2], [0, 0, 0], [0, 4, 5]],
              [[1, 4, 4], [1, 3, 1], [1, 2, 2], [1, 1, 2]],
              [[1, 1, 1], [1, 2, 2], [1, 3, 4], [1, 4, 4]]]


x=np.sum(np.array(xs), axis=0)
np.diagonal(x)
        [[sum(np.array(x)).tolist() for x in zip(*xs)]
np.count_nonzero(xs[0])


[[sum(term) for term in topic] for topic in xs]

[[sum(term) for term in topic] for topic in xs]

for i in range(3):
    for i+1 in range(4):
        lambda x,y:
    res = [(a, b) for idx, a in enumerate(t) for b in t[idx + 1:]]
    [a for a,b in res if a and b]
    for a,b in res

    def update_probability(self):
        for i in range(self.num_documents):
            for j in range(len(self.words_in_document[i])):
                word = self.words_in_document[i][j]
                for k in range(self.num_topics):
                    self.probability[k, word, i] = self.pheromone[k, word] * (1 / (sum([self.pheromone[l, word] for l in range(self.num_topics)])))

    def update_pheromone(self):
        delta_pheromone = np.zeros((self.num_topics, self.num_words))
        for i in range(self.num_documents):
            for j in range(len(self.words_in_document[i])):
                word = self.words_in_document[i][j]
                topic = self.topics_in_document[i][j]
                delta_pheromone[topic, word] += self.Q / len(self.words_in_document[i])
        self.pheromone = (1 - self.rho) * self.pheromone + delta_pheromone

    def select_topic(self, word, document):
        probability = self.probability[:, word, document]
        probability /= sum(probability)
        cum_probability = np.cumsum(probability)
        random_number = random.random()
        selected_topic = np.where(cum_probability >= random_number)[0][0]
        return selected_topic





    def run(self, num_iterations):
        for iteration in range(num_iterations):
            for ant in range(self.num_ants):
                for i in range(self.num_documents):
                    for j in range(len(self.words_in_document[i])):
                        word = self.words_in_document[i][j]
                        topic = self.select_topic(word, i)
                        self.topics_in_document[i][j] = topic
                self.update_pheromone()
            self.update_probability()

        return self.pheromone

    # Define the objective function to be optimized
    # doc_term_matrix is a matrix with size (num_unique_words, num_documents)
    def obj_func(term_doc_matrix, term_cat_matrix, cat_doc, alpha, beta):
        num_topics = cat_doc.shape[0]
        num_words, num_docs = term_doc_matrix.shape

        beta = 0.5
        alpha = 0.5

        # Calculate the log-likelihood
        log_likelihood = 0
        for i in range(num_docs):
            for j in range(num_words):
                prob_word_given_topic = (term_cat_matrix[j, :] + beta) / (np.sum(term_cat_matrix, axis=0) + num_words * beta)
                prob_topic_given_doc = (cat_doc[i, :] + alpha) / (np.sum(cat_doc, axis=0) + num_topics * alpha)
                log_likelihood += term_doc_matrix[j, i] * np.log(prob_word_given_topic.dot(prob_topic_given_doc))

        return log_likelihood


    # Define the ACO algorithm for topic modeling
    def aco_topic_modeling(term_doc_matrix, num_topics, num_ants, num_iterations, alpha, beta, rho, q0):
        num_words, num_docs = term_doc_matrix.shape
        num_topics = cat_doc.shape[0]

        # Initialize the pheromone matrix
        pheromone = np.ones((num_words, num_topics)) / num_topics

        # Initialize the best solution
        best_solution = np.zeros(num_docs, dtype=int)
        best_fitness = float('-inf')

        num_iterations = 10
        num_ants = 10
        q0 = 0.25

        # Run the ACO algorithm for num_iterations iterations
        for t in range(num_iterations):
            # Initialize the ant solutions
            ant_solutions = np.zeros((num_ants, num_docs), dtype=float)
    # asusx543ub@ad
            # Construct ant solutions
            for k in range(num_ants):
                # Initialize the ant solution
                ant_solution = np.zeros(num_docs, dtype=float)
                ant_solution[0] = np.random.randint(num_topics)

                # Build the ant solution
                for i in range(1, num_docs):
                    prob = np.zeros(num_topics)
                    for j in range(num_topics):
                        prob[j] = pheromone[:, j].dot(term_doc_matrix[:, i]) ** alpha * (
                                1.0 / (np.sum(pheromone[:, j]) + 1e-6)) ** beta

                    if np.random.rand() < q0:
                        ant_solution[i] = np.argmax(prob)
                    else:
                        ant_solution[i] = np.random.choice(num_topics, p=prob / np.sum(prob))

                ant_solutions[k] = ant_solution

            # Update the pheromone matrix
            delta_pheromone = np.zeros((num_words, num_topics))
            for k in range(num_ants):
                fitness = obj_func(term_doc_matrix, ant_solutions[k], alpha, beta)

                if fitness > best_fitness:
                    best_solution = ant_solutions[k].copy()
                    best_fitness = fitness

                for i in range(num_docs):
                    delta_pheromone[:, ant_solutions[k][i]]


