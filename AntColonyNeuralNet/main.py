import ReadDataset
import Utilities
import Sampling
import numpy as np
import os
import time
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models


df = ReadDataset.read_dataset(r'dataset/research/Train.csv')
test_df = ReadDataset.read_dataset(r'dataset/research/Test.csv')

df = df.set_index('id')
# test_df = test_df.set_index('id')

while True:
    ans = input('Remove saved data? Y/N\t')
    if ans == 'Y' or ans == 'N' or ans == 'n' or ans == 'y':
        break

if ans == 'Y' or ans == 'y':
    if Utilities.file_exists(r'outputs/requirements.npz'):
        requirements = os.path.join(r'outputs/requirements.npz')
        os.remove(requirements)
    if Utilities.file_exists(r'outputs/requirements.npz'):
        exploration_data = os.path.join(r'outputs/exploration_data.npz')
        os.remove(exploration_data)
    if Utilities.file_exists(r'outputs/requirements.npz'):
        iteration_data = os.path.join(r'outputs/iteration_data.npz')
        os.remove(iteration_data)
    print('All files removed successfully.')
else:
    print('No file removed!')

favorite_topics = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
favorite_topics_id = [0, 1, 2, 3]

# keeping only abstracts or texts
documents_df = df[['ABSTRACT']]
# test_documents_df = test_df[['ABSTRACT']]

# create a dataframe for labels
labels_df = df[favorite_topics]
# test_labels_df = test_df[favorite_topics]

# checks if there is any document with more than topic (label) only one is kept.
labels_df['selected_topic'] = labels_df.apply(lambda row: np.random.choice(np.nonzero(row.tolist())[0], size=1, replace=False).tolist(), axis=1)
# test_labels_df['selected_topic'] = test_labels_df.apply(lambda row: np.random.choice(np.nonzero(row)[0], size=1, replace=False)[0], axis=1)

# create a dictionary from the data frame for both the documents and topics.
fav_documents = dict(zip(documents_df.index.to_list(), documents_df.values.tolist()))
fav_documents_topics = dict(zip(labels_df.index.to_list(), labels_df['selected_topic'].tolist()))

# checks if we have saved samples or not. If YES, they are loaded and if NO, new samples are taken
sample_keys, train_doc_ids, test_doc_ids = Utilities.get_requirements()

# This part of code checks if there were any saved sample =====================
if sample_keys is not None:
    sample_documents = {key: fav_documents[key] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}
    print("Previous sample loaded!")
else:
    # take sample of documents for pre-study
    # sample_keys = ut.get_sample(list(fav_documents.keys()), 0.1)
    sample_keys = Sampling.get_sample_equal_size(0.2, favorite_topics_id, fav_documents_topics)
    sample_documents = {key: fav_documents[key][0] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}
    print("New samples taken!")

    # split dataset into train and test sets (we have their ids, in the next lines we retrieve documents and labels)
    train_doc_ids, test_doc_ids = Sampling.get_train_test_data(list(sample_documents.keys()), 0.8)

# This part of code takes sample of documents =====================
# Retrieve train docs
train_documents = {doc_id: sample_documents[doc_id] for doc_id in train_doc_ids}
train_topics = {doc_id: sample_labels[doc_id] for doc_id in train_doc_ids}

# Retrieve test docs
test_documents = {doc_id: sample_documents[doc_id] for doc_id in test_doc_ids}
test_topics = {doc_id: sample_labels[doc_id] for doc_id in test_doc_ids}
# ===================================================================================

tokenized_documents = Sampling.preprocess_manual(train_documents)
unique_tokens = set([term for doc in tokenized_documents.values() for term in doc])
tokenized_test_documents = Sampling.preprocess_manual(test_documents)
test_unique_tokens = set([term for doc in tokenized_test_documents.values() for term in doc])

# convert train dictionaries into np array as the number of labels for each document can be different.
tokenized_documents_arr = np.array(list(tokenized_documents.values()), dtype=object)
train_topics_arr = np.array(list(train_topics.values()), dtype=object)

# convert test dictionaries into np array as the number of labels for each document can be different.
tokenized_test_docs_arr = np.array(list(tokenized_test_documents.values()), dtype=object)
test_topics_arr = np.array(list(test_topics.values()), dtype=object)

# convert topics into topic_ids
train_topics_ids = [[favorite_topics_id.index(topic) for topic in doc_topic_set] for doc_topic_set in train_topics_arr]
test_topics_ids = [[favorite_topics_id.index(topic) for topic in doc_topic_set] for doc_topic_set in test_topics_arr]

training_topics_set = list(set(label for label_set in sample_labels.values() for label in label_set))

# ============== Check for Cosine Similarity of terms and reduce the number of terms(features) =============

cosine_sim_score = Utilities.get_cosine_similarity_score(tokenized_documents)

filtered_tokenized_documents = Utilities.filter_documents(tokenized_documents, cosine_sim_score)

# update the number of unique tokens after filtering using cosine similarity
unique_tokens = list(set(term for doc in filtered_tokenized_documents for term in doc))

# generate a dictionary of topics and all the terms under each
topics_docs_dic = {}

for doc, topic_set in zip(filtered_tokenized_documents, train_topics_ids):
    for topic in topic_set:
        if topic not in topics_docs_dic.keys():
            topics_docs_dic[topic] = []
        topics_docs_dic[topic].extend(doc)

# dictionary of concatenation of all documents for each topic.
for key in topics_docs_dic.keys():
    topics_docs_dic[key] = [" ".join(topics_docs_dic[key])]

# topics_docs contains len(topics) number of elements. Each element is a long string of all documents.
topics_docs = [doc[0] for doc in list(topics_docs_dic.values())]

vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_features=None, stop_words=None, tokenizer=lambda x: x.split())
tfidf_matrix = vectorizer.fit_transform(topics_docs)

num_topics = len(training_topics_set)
num_documents = len(tokenized_documents_arr)
num_words = len(unique_tokens)

# Create a dictionary of unique words
dictionary = corpora.Dictionary(tokenized_documents_arr)

# Convert the corpus into a bag-of-words representation. Same dictionary is used for both train and test.
# Words that are not in the dictionary from the test_docs are ignored.
corpus = [dictionary.doc2bow(document) for document in tokenized_documents_arr]
test_corpus = [dictionary.doc2bow(document) for document in tokenized_test_docs_arr]

# Define the ACO parameters
num_ants = 10
num_exploration = 20
num_iterations = 10
beta = 0.04  # initial percentage of taken from heuristic (global pheromone) - percentage changes through iterations
theta = 0.3  # effect of exploration ants pheromones (local)
evaporation_rate = 0.2
epsilon = 0.000001
lda_passes_count = 20
sliding_window = 10
load_exploration = False
exploring_ants = None

Utilities.set_requirements(np.array(sample_keys), np.array(train_doc_ids), np.array(test_doc_ids))

# save the best global solution (at each iteration)
iteration_start, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics = Utilities.get_iteration_data()

# Define the pheromone matrix size=(num_words, num_topics)
if pheromone_matrix is None:
    pheromone_matrix = np.array(tfidf_matrix.toarray())
    pheromone_matrix[pheromone_matrix == 0] += epsilon
    # pheromone_matrix = np.ones(shape=(num_topics, num_words))
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=0, norm='l1')

# updates beta if already some iteration had been passed. O.W. doesn't change value of beta.
beta = (iteration_start + 1) * beta

for iteration in range(iteration_start, num_iterations):
    start_time = time.time()

    exploration_best_solution = None
    exploration_best_solution_score = float('inf')
    exploration_best_metrics = None
    exploration_start = 0

    if load_exploration:
        exploration_start, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics = Utilities.get_exploration_data()
        if exploration_start >= num_exploration: exploration_start = 0
        load_exploration = False

    beta = beta + 0.05  # update pheromone effect percentage

    # generates the first group of ants here!
    if exploring_ants is None:
        exploring_ants = np.random.rand(num_ants, num_topics, num_words)

    for exploration_index in range(exploration_start, num_exploration):

        # keeps fitness function of generated ants
        exploring_ants_fitness_val = np.array([])

        # keeps exploring ants coverage, coherence, and perplexity
        exploring_ants_metrics = []

        exploring_topic_distribution = np.empty(shape=(num_ants, num_topics, num_words))

        # add the effect of global pheromone
        for ant_index in range(num_ants):
            exploring_topic_distribution[ant_index] = np.multiply(exploring_ants[ant_index] * (1-beta), pheromone_matrix * beta)  # pheromone and ant have different dimensions
            exploring_topic_distribution[ant_index][exploring_topic_distribution[ant_index] < epsilon] = epsilon
            exploring_topic_distribution[ant_index] = preprocessing.normalize(exploring_topic_distribution[ant_index], axis=0, norm='l1')

        for topic_distribution in exploring_topic_distribution:
            # Evaluate the quality of the new topic distribution
            new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto',
                                      eta=topic_distribution, passes=lda_passes_count)

            coverages = Utilities.calculate_coverage(num_documents, num_topics, num_words, new_lda, corpus)
            coverage_score = np.power(np.sum(np.power(coverages, 2)) / num_documents, 0.5)
            # coherence_score = models.CoherenceModel(model=new_lda, texts=tokenized_documents_arr, dictionary=dictionary, coherence='c_uci').get_coherence() * -1

            topic_tokens_dic = {}
            for topic_id, _ in enumerate(favorite_topics):
                topic_tokens_dic[topic_id] = new_lda.get_topic_terms(topic_id, sliding_window)
            coherence_score = Utilities.calculate_coherence(topic_tokens_dic, num_topics, sliding_window)

            test_doc_topics = {}  # keeps topics of each document
            test_topic_terms_dic = {}  # keeps the terms and their probabilities under each topic

            for doc_id, doc in enumerate(test_corpus):
                test_doc_topics[doc_id] = new_lda.get_document_topics(doc)

            for topic_id, _ in enumerate(favorite_topics):
                test_topic_terms_dic[topic_id] = new_lda.get_topic_terms(topic_id, num_words)

            topics_terms_list = list(test_topic_terms_dic.values())
            topics_terms_list = [[list(term_prob_pair) for term_prob_pair in term_prob_list] for topic_id, term_prob_list in enumerate(topics_terms_list)]
            perplexity_score = Utilities.calculate_perplexity(test_doc_topics, topics_terms_list, test_corpus)

            objective_metrics = (coverage_score, coherence_score, perplexity_score)
            theoretical_best_solution = (0, 0, 0)
            objective_value = np.sqrt(np.sum(np.square(np.subtract(objective_metrics, theoretical_best_solution))))
            # saves the fitness value of the ant
            exploring_ants_fitness_val = np.append(exploring_ants_fitness_val, objective_value)
            # saves objective metrics
            exploring_ants_metrics.append(objective_metrics)

        # Update the best solution found in this exploration
        if np.min(exploring_ants_fitness_val) < exploration_best_solution_score:
            exploration_best_metrics = exploring_ants_metrics[exploring_ants_fitness_val.argmin()]
            exploration_best_solution = exploring_topic_distribution[exploring_ants_fitness_val.argmin()]
            exploration_best_solution_score = np.min(exploring_ants_fitness_val)

        # remained pheromones from the previous ants are evaporated by the evaporation rate
        exploration_best_solution *= (1-evaporation_rate)

        # selected leader ants which their pheromone trace is used
        selected_leader_ants = Utilities.roulette_wheel(exploring_ants_fitness_val)

        # generate new ants
        exploring_ants = np.random.rand(num_ants, num_topics, num_words)

        # include local pheromone in new ants path
        for ant_index in range(num_ants):
            exploring_ants[ant_index] = np.multiply(exploring_ants[ant_index] * (1-theta), exploring_topic_distribution[selected_leader_ants[ant_index]] * theta)
            exploring_ants[ant_index] = preprocessing.normalize(exploring_ants[ant_index], axis=0, norm='l1')

        Utilities.save_exploration_data(exploration_index+1, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics)

    if exploration_best_solution_score < best_solution_score:
        best_iteration = iteration
        best_metrics = exploration_best_metrics
        best_solution = exploration_best_solution
        best_solution_score = exploration_best_solution_score

    # Update the pheromone matrix
    pheromone_matrix = np.add(pheromone_matrix, exploration_best_solution)
    pheromone_matrix *= (1 - evaporation_rate)
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=0, norm='l1')

    Utilities.save_iteration_data(iteration+1, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics)
