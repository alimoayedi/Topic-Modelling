import numpy as np
import random

class AntColony:
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

    def add_document(self, document):
        self.documents.append(document)
        words = document.split()
        self.words_in_document.append(words)
        self.topics_in_document.append(np.zeros(len(words), dtype=np.int))

    def update_probability(self):
        for i in range(self.num_documents):
            for j in range(len(self.words_in_document[i])):
                word = self.words_in_document[i][j]
                for k in range(self.num_topics):
                    self.probability[k, word, i] = self.pheromone[k, word] * (1 / (sum([self.pheromone[l, word] for l in range(self.num_topics)])))

    def select_topic(self, word, document):
        probability = self.probability[:, word, document]
        probability /= sum(probability)
        cum_probability = np.cumsum(probability)
        random_number = random.random()
        selected_topic = np.where(cum_probability >= random_number)[0][0]
        return selected_topic

    def update_pheromone(self):
        delta_pheromone = np.zeros((self.num_topics, self.num_words))
        for i in range(self.num_documents):
            for j in range(len(self.words_in_document[i])):
                word = self.words_in_document[i][j]
                topic = self.topics_in_document[i][j]
                delta_pheromone[topic, word] += self.Q / len(self.words_in_document[i])
        self.pheromone = (1 - self.rho) * self.pheromone + delta_pheromone

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
