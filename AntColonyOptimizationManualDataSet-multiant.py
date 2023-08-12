import numpy as np
from sklearn import preprocessing
from gensim import corpora, models
from random import uniform

import multiprocessing

# Local classes
from LoadDataset import LoadReutersDataset
import utilities as ut

import pdb

# load data
dataset_path = 'C:/Thesis/Dataset/reuters21578'
loader = LoadReutersDataset(data_path=dataset_path)
documents_dic, topics_dic, _, _, _, _, _ = loader.load()

# find docs without any topic
doc_ids_without_topic = [doc_id for doc_id in documents_dic.keys() if len(topics_dic[doc_id]) == 0]

# filter out documents without any topic
documents = {key: value for key, value in documents_dic.items() if key not in doc_ids_without_topic}
documents_topics = {key: value for key, value in topics_dic.items() if key not in doc_ids_without_topic}

favorite_topics = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

# filter documents and keep only the ones with favorite topics
fav_documents_topics = {doc_id: topic_list for doc_id, topic_list in documents_topics.items() if any(topic in topic_list for topic in favorite_topics)}

fav_documents = {doc_id: documents[doc_id] for doc_id in fav_documents_topics.keys()}

# remove the topics from the documents topics that are not in the favorite list
for doc_id in fav_documents_topics.keys():
    fav_documents_topics[doc_id] = [topic for topic in fav_documents_topics[doc_id] if topic in favorite_topics]

# checks if we have saved samples or not. If YES, they are loaded and if NO, new samples are taken
if ut.file_exists('sample_keys') and ut.file_exists('train_doc_ids') and ut.file_exists('test_doc_ids'):
    sample_keys = np.load('C:/Thesis/saved_status/sample_keys.npy')
    train_doc_ids = np.load('C:/Thesis/saved_status/train_doc_ids.npy')
    test_doc_ids = np.load('C:/Thesis/saved_status/test_doc_ids.npy')

    sample_documents = {key: fav_documents[key] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}
else:
    # take sample of documents for pre-study
    sample_keys = ut.get_sample(list(fav_documents.keys()), 0.1)
    sample_documents = {key: fav_documents[key] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}

    # split dataset into train and test sets (we have their ids, in the next lines we retrieve documents and labels)
    train_doc_ids, test_doc_ids = ut.get_train_test_data(list(sample_documents.keys()), 0.8)

# ================= This part of code takes sample of documents =====================
# Retrieve train docs
train_documents = {doc_id: sample_documents[doc_id] for doc_id in train_doc_ids}
train_topics = {doc_id: sample_labels[doc_id] for doc_id in train_doc_ids}

# Retrieve test docs
test_documents = {doc_id: sample_documents[doc_id] for doc_id in test_doc_ids}
test_topics = {doc_id: sample_labels[doc_id] for doc_id in test_doc_ids}
# ===================================================================================

tokenized_documents, unique_tokens = ut.preprocess_manual(train_documents)
tokenized_test_documents, test_unique_tokens = ut.preprocess_manual(test_documents)

# convert train dictionaries into np array as the number of labels for each document can be different.
tokenized_documents_arr = np.array(list(tokenized_documents.values()), dtype=object)
train_topics_arr = np.array(list(train_topics.values()), dtype=object)

# convert test dictionaries into np array as the number of labels for each document can be different.
tokenized_test_docs_arr = np.array(list(tokenized_test_documents.values()), dtype=object)
test_topics_arr = np.array(list(test_topics.values()), dtype=object)

# convert topics into topic_ids
train_topics_ids = [[favorite_topics.index(topic) for topic in doc_topic_set] for doc_topic_set in train_topics_arr]
test_topics_ids = [[favorite_topics.index(topic) for topic in doc_topic_set] for doc_topic_set in test_topics_arr]

training_topics_set = list(set(label for label_set in sample_labels.values() for label in label_set))

num_topics = len(training_topics_set)
num_documents = len(tokenized_documents_arr)
num_words = len(unique_tokens)

# Create a dictionary of unique words
dictionary = corpora.Dictionary(tokenized_documents_arr)

# Convert the corpus into a bag-of-words representation. Same dictionary is used for both train and test.
# Words that are not in the dictionary from the test_docs are ignored.
corpus = [dictionary.doc2bow(document) for document in tokenized_documents_arr]
test_corpus = [dictionary.doc2bow(document) for document in tokenized_test_docs_arr]

# returns the topic distribution and the frequency of each topic. Used for plotting.
# ut.get_topics_distribution(training_topics_set, train_topics_ids)  # training data
# ut.get_topics_distribution(training_topics_set, test_topics_ids)  # test data
# ut.plot_topic_distribution(training_topics_set, train_topics_ids)

# Define the ACO parameters
num_ants = 10
num_exploration = 30
num_iterations = 10
beta = 0.1  # initial percentage of taken from heuristic (global pheromone) - percentage changes through iterations
theta = 0.3  # effect of exploration ants pheromones (local)
evaporation_rate = 0.2

# Define the pheromone matrix size=(num_words, num_topics)
if ut.file_exists('pheromone_matrix'):
    pheromone_matrix = np.load('C:/Thesis/saved_status/pheromone_matrix.npy')
else:
    pheromone_matrix = np.ones(shape=(num_topics, num_words))
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=0, norm='l1')

# save the best global solution (at each iteration)
best_solution = None
best_solution_score = float('inf')

np.save('C:/Thesis/saved_status/sample_keys.npy', np.array(sample_keys))
np.save('C:/Thesis/saved_status/train_doc_ids.npy', np.array(train_doc_ids))
np.save('C:/Thesis/saved_status/test_doc_ids.npy', np.array(test_doc_ids))

if ut.file_exists('iteration'):
    iteration_start = int(np.load('C:/Thesis/saved_status/iteration.npy'))
else:
    iteration_start = 0

for iteration in range(iteration_start, num_iterations):
    # save the best local solution at each exploration step
    exploration_best_solution = None
    exploration_best_solution_score = float('inf')

    beta = beta + (iteration * 0.5)  # update pheromone effect percentage

    # generates the first group of ants here!
    exploring_ants = np.random.rand(num_ants, num_topics, num_words)

    for exploration_index in range(num_exploration):

        # keeps fitness function of generated ants
        exploring_ants_fitness_val = np.array([])

        exploring_topic_distribution = np.empty(shape=(num_ants, num_topics, num_words))

        # add the effect of global pheromone
        for ant_index in range(num_ants):
            exploring_topic_distribution[ant_index] = np.multiply(exploring_ants[ant_index] * (1-beta), pheromone_matrix * beta)  # pheromone and ant have different dimensions
            exploring_topic_distribution[ant_index] = preprocessing.normalize(exploring_topic_distribution[ant_index], axis=0, norm='l1')

        for topic_distribution in exploring_topic_distribution:
            # Evaluate the quality of the new topic distribution
            new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto',
                                      eta=topic_distribution)

            coverages = ut.calculate_coverage(num_documents, num_topics, num_words, new_lda, corpus)
            coverage_score = np.power(sum(np.power(coverages, 2)) / num_documents, 0.5)
            coherence_score = models.CoherenceModel(model=new_lda, texts=tokenized_documents_arr,
                                                    dictionary=dictionary,
                                                    coherence='c_uci').get_coherence() * -1

            # calculate perplexity
            docs_perplexity = []  # keeps perplexity for each doc
            for doc_index, doc in enumerate(corpus):
                log_likelihood = 0
                # gets topic and their probabilities for each doc
                test_doc_topics = new_lda.get_document_topics(doc)
                doc_length = len(doc)
                for word_id, word_count in doc:  # loop over words
                    word_topic_probability = 0
                    for topic_id, topic_prob in test_doc_topics:
                        # gets words per topic from model
                        for word, prob in new_lda.get_topic_terms(topic_id):
                            word_prob = prob if word_id == word else 1e-6
                        word_topic_probability += word_prob * topic_prob  # probability of word over all topics
                    if word_topic_probability == 0:
                        pdb.set_trace()
                    log_likelihood += word_count * np.log2(word_topic_probability)

                # -inf is replaced with 0, since -inf means no prediction was possible based on the trained model
                if log_likelihood == -np.inf:
                    log_likelihood = 0
                docs_perplexity.append(log_likelihood)
                # docs_perplexity.append(np.power(2, -1 * log_likelihood / doc_length))

            total_terms_in_test_docs = sum([count for doc in test_corpus for _, count in doc])
            perplexity_score = -1 * sum(docs_perplexity) / (10 * total_terms_in_test_docs)
            # perplexity_score = sum(docs_perplexity) / len(docs_perplexity)

            objective_values = (coverage_score, coherence_score, perplexity_score)
            theoretical_best_solution = (0, 0, 0)
            objective_value = np.sqrt(np.sum(np.square(np.subtract(objective_values, theoretical_best_solution))))
            # saves the fitness value of the ant
            exploring_ants_fitness_val = np.append(exploring_ants_fitness_val, objective_value)
            if objective_value == np.inf:
                pdb.set_trace()
            if True in np.isnan(exploring_ants_fitness_val):
                pdb.set_trace()

        # Update the best solution found in this exploration
        if np.min(exploring_ants_fitness_val) < exploration_best_solution_score:
            exploration_best_solution = exploring_topic_distribution[exploring_ants_fitness_val.argmin()]
            exploration_best_solution_score = np.min(exploring_ants_fitness_val)

        # remained pheromones from the previous ants are evaporated by the evaporation rate
        exploring_ants_fitness_val *= (1-evaporation_rate)

        # selected leader ants which their pheromone trace is used
        selected_leader_ants = ut.roulette_wheel(exploring_ants_fitness_val)

        # generate new ants
        exploring_ants = np.random.rand(num_ants, num_topics, num_words)

        # include local pheromone in new ants path
        for ant_index in range(num_ants):
            exploring_ants[ant_index] = np.multiply(exploring_ants[ant_index] * (1-theta), exploring_topic_distribution[selected_leader_ants[ant_index]] * theta)
            exploring_ants[ant_index] = preprocessing.normalize(exploring_ants[ant_index], axis=0, norm='l1')

    if exploration_best_solution_score < best_solution_score:
        best_solution = exploration_best_solution
        best_solution_score = exploration_best_solution_score

    # Update the pheromone matrix
    pheromone_matrix = np.add(pheromone_matrix, exploration_best_solution)
    pheromone_matrix *= (1 - evaporation_rate)
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=0, norm='l1')

    print("iteration: ", iteration, " coherence", coherence_score, " coverage: ", coverage_score, " preplexity: ", perplexity_score)

    np.save('C:/Thesis/saved_status/iteration.npy', iteration)
    np.save('C:/Thesis/saved_status/pheromone_matrix.npy', pheromone_matrix)

    # Print the best solution found so far
    print('Iteration', iteration, 'Best objective function:', best_solution_score, 'Best solution:', best_solution)