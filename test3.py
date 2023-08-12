import numpy as np
import random


class AntColony:
    def _init_(self, num_topics, num_words, num_ants, num_iterations, alpha, beta, rho, Q):
        """
        Initializes an instance of the AntColony class.

        Parameters:
        num_topics (int): the number of topics to be modeled
        num_words (int): the size of the vocabulary
        num_ants (int): the number of ants to be used in the ACO algorithm
        num_iterations (int): the number of iterations to run the ACO algorithm for
        alpha (float): the relative importance of the pheromone trail when selecting a topic
        beta (float): the relative importance of the topic probability when selecting a topic
        rho (float): the rate at which the pheromone evaporates
        Q (float): the amount of pheromone to be deposited by each ant
        """
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone_matrix = np.ones((num_topics, num_words)) / num_topics
        self.document_topic_matrix = np.zeros((len(documents), num_topics))
        self.word_topic_matrix = np.zeros((num_words, num_topics))

        # initialize word_topic_matrix
        # normalize the count of a given word assigned to a particular topic across all documents in the corpus.
        # without normalization, longer documents would have a greater influence on the word-topic matrix,
        # since they contain more words and therefore more counts of each word
        for i, doc in enumerate(documents):
            bow_doc = self.convert_to_bow(doc, vocabulary)
            for j, word_count in enumerate(bow_doc):
                for k in range(num_topics):
                    self.word_topic_matrix[j, k] += word_count / len(documents)

        # normalize word_topic_matrix
        self.word_topic_matrix /= np.sum(self.word_topic_matrix, axis=0)

    def run(self, documents, vocabulary):
        """
        Runs the ACO algorithm for topic modeling.

        Parameters:
        documents (list): a list of documents, where each document is a list of words
        vocabulary (list): a list of words in the vocabulary

        Returns:
        document_topic_matrix (numpy.ndarray): a matrix where each row represents a document and each column represents a topic,
        and the value at (i, j) is the number of times topic j appears in document i
        word_topic_matrix (numpy.ndarray): a matrix where each row represents a word in the vocabulary and each column represents a topic,
        and the value at (i, j) is the number of times word i appears in documents assigned to topic j
        """
        bow_documents = [self.convert_to_bow(document, vocabulary) for document in documents]
        ant_solutions = [self.initialize_ant_solution(bow_document) for bow_document in bow_documents]

        for iteration in range(self.num_iterations):
            objective_values = []
            for ant_solution in ant_solutions:
                coverage = self.calculate_coverage(ant_solution)
                objective_values.append(coverage)
                self.update_pheromone_matrix(ant_solution, coverage)

            best_ant_solution = ant_solutions[np.argmax(objective_values)]
            self.update_document_topic_matrix(best_ant_solution, bow_documents)
            self.update_word_topic_matrix(best_ant_solution, vocabulary)
            ant_solutions = [self.construct_ant_solution(bow_document) for bow_document in bow_documents]

        return self.document_topic_matrix, self.word_topic_matrix

    def initialize_ant_solution(self, bow_document):
        """
        Initializes an ant solution by randomly assigning topics to words in a document.

        Parameters:
        bow_document (list): a bag of words representation of a document

        Returns:
        ant_solution (list): a list where each element is an integer representing a topic assigned to a word in the document
        """
        return [random.randint(0, self.num_topics - 1) for i in range(len(bow_document))]

def construct_ant_solution(self, bow_document):
    """
    Constructs an ant solution by iteratively selecting a topic for each word in a document.

    Parameters:
    bow_document (list): a bag of words representation of a document

    Returns:
    ant_solution (list): a list where each element is an integer representing a topic assigned to a word in the document
    """
    ant_solution = [-1] * len(bow_document)

    for i in range(len(bow_document)):
        word_index = bow_document[i]
        topic_probabilities = self.calculate_topic_probabilities(word_index)
        pheromone_probabilities = self.pheromone_matrix[word_index] ** self.alpha
        probabilities = (1 - self.beta) * pheromone_probabilities + self.beta * topic_probabilities
        probabilities /= sum(probabilities)
        ant_solution[i] = np.random.choice(np.arange(self.num_topics), p=probabilities)

    return ant_solution


def calculate_topic_probabilities(self, word_index):
    """
    Calculates the probability of each topic being assigned to a given word.

    Parameters:
    word_index (int): the index of the word in the vocabulary

    Returns:
    topic_probabilities (numpy.ndarray): a 1D array where each element represents the probability of a topic being assigned to the word
    """
    topic_probabilities = np.zeros(self.num_topics)
    for j in range(self.num_topics):
        topic_probabilities[j] = self.word_topic_matrix[word_index, j] / np.sum(self.word_topic_matrix[:, j])
    return topic_probabilities


def calculate_coverage(self, ant_solution):
    """
    Calculates the coverage of an ant solution, which is the number of distinct topics used in the solution.

    Parameters:
    ant_solution (list): a list where each element is an integer representing a topic assigned to a word in the document

    Returns:
    coverage (int): the number of distinct topics used in the solution
    """
    return len(set(ant_solution))


def update_pheromone_matrix(self, ant_solution, coverage):
    """
    Updates the pheromone matrix based on the ant solution and the coverage of the solution.

    Parameters:
    ant_solution (list): a list where each element is an integer representing a topic assigned to a word in the document
    coverage (int): the number of distinct topics used in the solution
    """
    for i in range(len(ant_solution)):
        self.pheromone_matrix[ant_solution[i], bow_document[i]] += self.Q / coverage
    self.pheromone_matrix *= (1 - self.rho)


def convert_to_bow(self, document, vocabulary):
    """
    Converts a document to a bag of words representation.

    Parameters:
    document (list): a list of words in the document
    vocabulary (list): a list of words in the vocabulary

    Returns:
    bow_document (list): a list where each element is an integer representing the index of a word in the vocabulary
    """
    bow_document = [0] * len(vocabulary)
    for word in document:
        if word in vocabulary:
            bow_document[vocabulary.index(word)] += 1
    return bow_document


def update_document_topic_matrix(self, ant_solution, bow_documents):
    """
    Updates the document topic matrix based on the best ant solution found in an iteration.

    Parameters:
    ant_solution (list): a list where each element is an integer representing a topic assigned to a word in the document