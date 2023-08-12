import nltk

nltk.download('reuters')
from nltk.corpus import reuters


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(doc):
    tokens = word_tokenize(doc.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

docs = [preprocess(reuters.raw(doc_id)) for doc_id in reuters.fileids()]


from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel

dictionary = Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

def coverage(topics):
    doc_topic = [max(t, key=lambda x: x[1])[0] for t in topics]
    return len(set(doc_topic)) / len(doc_topic)

def coherence(topics):
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='c_v')
    return cm.get_coherence()

def perplexity(topics):
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=len(topics))
    return lda.log_perplexity(corpus_tfidf)


'''
Finally, we can define our ant colony optimization algorithm. In this example, we will use the ACO algorithm to find the optimal number of topics for the LDA model. We will use three types of pheromones: global pheromones (used to guide the overall search), local pheromones (used to guide the local search), and personal pheromones (used to remember the best solution found by each ant):
'''

import random
import numpy as np

class Ant:
    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.topics = random.sample(range(1, dictionary.num_docs), num_topics)
        self.global_pheromone = np.ones(num_topics)
        self.local_pheromone = np.ones(num_topics)
        self.personal_pheromone = np.zeros(num_topics)
        self.coverage = 0
        self.coherence = 0
        self.perplexity = float('inf')

    def evaluate(self):
        topics = [docs[i] for i in self.topics]
        self.coverage = coverage(topics)
        self.coherence = coherence(topics)
        self.perplexity = perplexity(topics)

    def update_pheromones(self, global_pheromone, decay_rate, Q):
        # Update global pheromones
        for i in range(self.num_topics):
            self.global_pheromone[i] += global_pheromone[i]
        
        # Update local pheromones
        for i in range(self.num_topics):
            self.local_pheromone[i] = (1 - decay_rate) * self.local_pheromone[i] + decay_rate * self.global_pheromone[i]
        
        # Update personal pheromones
        if self.coverage > 0:
            delta_pheromone = Q / self.coverage
            for i in range(self.num_topics):
                self.personal_pheromone[i] += delta_pheromone
        
    def choose_next_topic(self, alpha, beta):
        # Calculate probability of choosing each topic
        prob = np.zeros(self.num_topics)
        for i in range(self.num_topics):
            prob[i] = (self.local_pheromone[i] ** alpha) * ((1 / (self.personal_pheromone[i] + 1)) ** beta)
        prob /= prob.sum()
        
        # Choose next topic based on probability
        next_topic = np.random.choice(range(self.num_topics), p=prob)
        self.topics[next_topic] = random.randint(1, dictionary.num_docs)
        
    def __str__(self):
        return f'Ant with topics {self.topics} (coverage: {self.coverage}, coherence: {self.coherence}, perplexity: {self.perplexity})'


def aco(num_ants, num_iterations, num_topics, alpha, beta, rho, Q):
    # Initialize pheromone matrix
    global_pheromone = np.ones(num_topics)
    
    # Initialize ants
    ants = [Ant(num_topics) for _ in range(num_ants)]
    
    # Run ACO iterations
    best_ant = None
    for i in range(num_iterations):
        # Evaluate ants
        for ant in ants:
            ant.evaluate()
            if best_ant is None or ant.coverage > best_ant.coverage:
                best_ant = ant
        
        # Update pheromones
        for ant in ants:
            ant.update_pheromones(global_pheromone, rho, Q)
        
        # Choose next topics for ants
        for ant in ants:
            ant.choose_next_topic(alpha, beta)
        
    # Return best solution
    return best_ant
