from gensim import corpora, models
from random import uniform
import Preprocess
import numpy as np

# Load the Reuters dataset
from nltk.corpus import reuters


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
    return index-1 if index-1 != 0 else 1


def calculate_coverage(num_documents, num_topics, num_words, model, corpus):
    n_tk = np.zeros(shape=num_topics)
    proportions = np.zeros(shape=num_topics)
    coverages = np.zeros(shape=num_documents)

    for topic in range(num_topics):
        n_tk[topic] = get_number_words_in_topic(model.get_topic_terms(topic, num_words))

    for doc in range(num_documents):
        doc_topics, word_topics, _ = model.get_document_topics(bow=corpus[doc], per_word_topics=True)
        doc_topics = [topic_id for topic_id, _ in doc_topics]
        for topic in doc_topics:
            sum_words_topic_tf = 0
            document_topic_num_words = 0
            for word_id, word_topic_list in word_topics:
                if topic in word_topic_list:
                    document_topic_num_words += 1
                    sum_words_topic_tf += [item[1] for item in corpus[doc] if item[0] == word_id][0]
                    print('document_topic_num_words:', document_topic_num_words)
            proportions[topic] = sum_words_topic_tf / (n_tk[topic] - document_topic_num_words + 1)

        under_root = 0
        total_tf = sum(freq for _, freq in corpus[doc])
        for word_id, word_tf in corpus[doc]:
            p1 = word_tf / total_tf
            p2 = 0
            for topic in doc_topics:
                p2 += (word_tf / sum_words_topic_tf) * proportions[topic]
            under_root += np.power(p1 - p2, 2)
        coverages[doc] = np.power(under_root, 0.5)
    return coverages


def main():
    # raw data and categories
    documents = reuters.fileids()
    train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
    categories = reuters.categories()
    train_docs = train_docs

    # number of dimensions
    documents_topics, tokenized_documents, unique_tokens = Preprocess.preprocess(categories, train_docs)
    num_topics = len(categories)
    num_documents = len(tokenized_documents)
    num_words = len(unique_tokens)

    # Create a dictionary of unique words
    dictionary = corpora.Dictionary(tokenized_documents)

    # Convert the corpus into a bag-of-words representation
    corpus = [dictionary.doc2bow(document) for document in tokenized_documents]

    # Set the number of topics
    num_topics = 90

    # Define the coherence metric
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # lda.get_topic_terms(1, topn=num_words)
    # lda.get_document_topics(bow=corpus[0], minimum_probability=None, per_word_topics=True)
    # lda.get_topics()
    #

    coverages = calculate_coverage(num_documents, num_topics, num_words, lda, corpus)
    coverage_model = np.power(sum(np.power(coverages, 2)) / num_documents, 0.5)

    # Define the coherence metric
    coherence_model = models.CoherenceModel(model=lda, texts=tokenized_documents, dictionary=dictionary, coherence='c_uci')
    preplexity_model = lda.log_perplexity(corpus)

    # Define the pheromone matrix size=(num_words, num_topics)
    pheromone_matrix = [[1.0] * num_topics for _ in range(len(dictionary))]

    # Define the ACO parameters
    num_ants = 50
    num_iterations = 100
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1

    # Initialize the ant population
    ants = []
    for i in range(num_ants):
        ant = [uniform(0, 1) for _ in range(num_words)]  # probability of each topic
        ants.append(ant)

    # Run the ACO algorithm
    best_solution = None
    best_solution_score = float('inf')

    for iteration in range(num_iterations):
        for ant in ants:
            # Generate a new topic distribution for the documents
            topic_distribution = ant

            # Evaluate the quality of the new topic distribution
            new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto', eta=topic_distribution)

            new_coverages = calculate_coverage(num_documents, num_topics, num_words, lda, corpus)
            new_coverage_model = np.power(sum(np.power(new_coverages, 2)) / num_documents, 0.5)
            new_coherence_model = models.CoherenceModel(model=new_lda, texts=tokenized_documents, dictionary=dictionary, coherence='c_uci').get_coherence() * -1
            new_preplexity_model = new_lda.log_perplexity(corpus)

            objective_value = (new_coverage_score, new_coherence_score, new_preplexity)
            theoretical_best_solution = (0, 0, 0)
            summation_square = np.sum(np.square(coverage_coherence_pair - theoretical_best_solution), axis=1)
            objective_values = np.sqrt(summation_square)

            # Update the pheromone matrix
            for word_id in range(len(dictionary)):
                for topic_id in range(num_topics):
                    pheromone_matrix[word_id][topic_id] *= (1 - evaporation_rate)
                    if topic_distribution[word_id] == topic_id:
                        pheromone_matrix[word_id][topic_id] += objective_values

            # Update the best solution found so far
            if objective_values < best_solution_score:
                best_solution = topic_distribution
                best_solution_score = new_coherence_score

            for word_id in range(len(dictionary)):
                topic_probabilities = []
                for topic_id in range(num_topics):
                    pheromone_value = pheromone_matrix[word_id][topic_id]
                    term_probability = lda.get_term_topics(word_id)[0][1] if len(lda.get_term_topics(word_id))!=0 else 0
                    topic_probability = (pheromone_value ** alpha) * ((term_probability + beta) ** beta)
                    topic_probabilities.append(topic_probability)
                topic_distribution.append(topic_probabilities.index(max(topic_probabilities)))


            print("ant: ", ant, " coherence", new_coherence_score, " coverage: ", new_coverage_score, " preplexity: ", new_preplexity)
        # Print the best solution found so far
        print('Iteration', iteration, 'Best coherence score:', best_solution_score, 'Best solution:', best_solution)

main()
