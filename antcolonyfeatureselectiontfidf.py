import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from random import uniform
import time
import math
import os

# Local classes
from LoadDataset import LoadReutersDataset
import utilities as ut

loader = LoadReutersDataset(data_path=r'dataset/reuters21578')
documents_dic, topics_dic, _, _, _, _, _ = loader.load()

prerequirments = os.path.join(r'outputs/prerequirments.npz')
exploration_data = os.path.join(r'outputs/exploration_data.npz')
iteration_data = os.path.join(r'outputs/iteration_data.npz')

while True:
    ans = input('Remove saved data? Y/N\t')
    if ans == 'Y' or ans == 'N' or ans == 'n' or ans == 'y':
        break

if ans == 'Y' or ans == 'y':
    os.remove(prerequirments)
    os.remove(exploration_data)
    os.remove(iteration_data)
    print('All files removed successfully.')
else:
    print('No file removed!')


def file_exists(filename):
    """Check if a file exists in the current directory"""
    directory = f'/outputs/{filename}.npz'
    return os.path.exists(directory) and os.path.isfile(directory)


def save_prerequirments(sample_key, train_doc_ids, test_doc_ids):
    dir = 'outputs/prerequirments.npz'
    np.savez(dir, sample_key=sample_key, train_doc_ids=train_doc_ids, test_doc_ids=test_doc_ids)


def get_prerequirments():
    if not file_exists('prerequirments'):
        return None, None, None
    data = np.load('/outputs/prerequirments.npz')
    sample_key = data['sample_key']
    train_doc_ids = data['train_doc_ids']
    test_doc_ids = data['test_doc_ids']
    return sample_key, train_doc_ids, test_doc_ids

def save_exploration_data(next_exploration_index, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics):
    dir = '/outputs/exploration_data.npz'
    np.savez(dir,
             next_exploration_index=next_exploration_index,
             exploring_ants=exploring_ants,
             exploration_best_solution=exploration_best_solution,
             exploration_best_solution_score=exploration_best_solution_score,
             exploration_best_metrics=exploration_best_metrics)

def get_exploration_data():
    if not file_exists('exploration_data'):
        return 0, None, None, float('inf'), None

    data = np.load('/outputs/exploration_data.npz')
    next_exploration_index = int(data['next_exploration_index'])
    exploring_ants = data['exploring_ants']
    exploration_best_solution = data['exploration_best_solution']
    exploration_best_solution_score = int(data['exploration_best_solution_score'])
    exploration_best_metrics = data['exploration_best_metrics']

    return next_exploration_index, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics

def save_iteration_data(Next_iteration_index, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics):
    dir = '/outputs/iteration_data.npz'
    np.savez(dir,
             Next_iteration_index=Next_iteration_index,
             pheromone_matrix=pheromone_matrix,
             best_iteration=best_iteration,
             best_solution=best_solution,
             best_solution_score=best_solution_score,
             best_metrics=best_metrics)

def get_iteration_data():
    if not file_exists('iteration_data'):
        return 0, None, 0, None, float('inf'), None

    data = np.load('/outputs/iteration_data.npz')
    Next_iteration_index = int(data['Next_iteration_index'])
    pheromone_matrix = data['pheromone_matrix']
    best_iteration = int(data['best_iteration'])
    best_solution = data['best_solution']
    best_solution_score = int(data['best_solution_score'])
    best_metrics = data['best_metrics']
    return Next_iteration_index, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics

# find docs without any topic
doc_ids_without_topic = [doc_id for doc_id in documents_dic.keys() if len(topics_dic[doc_id]) == 0]

# filter out documents without any topic
documents = {key: value for key, value in documents_dic.items() if key not in doc_ids_without_topic}
documents_topics = {key: value for key, value in topics_dic.items() if key not in doc_ids_without_topic}

favorite_topics = ['acq', 'corn', 'crude', 'earn']
#favorite_topics = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

# filter documents and keep only the ones with favorite topics
fav_documents_topics = {doc_id: topic_list for doc_id, topic_list in documents_topics.items() if any(topic in topic_list for topic in favorite_topics)}

fav_documents = {doc_id: documents[doc_id] for doc_id in fav_documents_topics.keys()}

# remove the topics from the documents topics that are not in the favorite list
for doc_id in fav_documents_topics.keys():
    fav_documents_topics[doc_id] = [topic for topic in fav_documents_topics[doc_id] if topic in favorite_topics]

# checks if we have saved samples or not. If YES, they are loaded and if NO, new samples are taken
sample_keys, train_doc_ids, test_doc_ids = get_prerequirments()

if sample_keys is not None:
    sample_documents = {key: fav_documents[key] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}
    print("Previous sample loaded!")
else:
    # take sample of documents for pre-study
    # sample_keys = ut.get_sample(list(fav_documents.keys()), 0.1)
    sample_keys = ut.get_sample_equal_size(0.2, favorite_topics, fav_documents_topics)
    sample_documents = {key: fav_documents[key] for key in sample_keys}
    sample_labels = {key: fav_documents_topics[key] for key in sample_keys}
    print("New samples taken!")

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

tokenized_documents = ut.preprocess_manual(train_documents)
unique_tokens = set([term for doc in tokenized_documents.values() for term in doc])
tokenized_test_documents = ut.preprocess_manual(test_documents)
test_unique_tokens = set([term for doc in tokenized_test_documents.values() for term in doc])

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

# ============== Check for Cosine Similarity of terms and reduce the number of terms(features) =============

joined_docs = [" ".join(words) for words in tokenized_documents.values()]

vectorizer = TfidfVectorizer()
vectorized_docs = vectorizer.fit_transform(joined_docs)
vectorized_docs = vectorized_docs.transpose()
terms_features = list(vectorizer.vocabulary_.keys())

cosine_sim_score = cosine_similarity(vectorized_docs)

filtered_tokenized_documents = []

for doc in tokenized_documents.values():
    to_remove = set()
    for item_1 in range(len(doc)):
        for item_2 in range(item_1 + 1, len(doc)):
            if cosine_sim_score[item_1, item_2] > 0.9:
                to_remove.add(doc[item_1])
    filtered_tokenized_documents.append([term for term in doc if term not in to_remove])

tokenized_documents_arr = filtered_tokenized_documents
unique_tokens = list(set(term for doc in filtered_tokenized_documents for term in doc))

# generate a dictionary of topics and all the terms under each
topics_docs_dic = {}

for doc, topic_set in zip(tokenized_documents_arr, train_topics_ids):
  for topic in topic_set:
    if topic not in topics_docs_dic.keys(): topics_docs_dic[topic] = []
    topics_docs_dic[topic].extend(doc)

# dictionary of concatination of all documents for each topic.
for key in topics_docs_dic.keys():
  topics_docs_dic[key] = [" ".join(topics_docs_dic[key])]

# topics_docs contains len(topics) number of elements. Each element is a long string of all documents.
topics_docs=[doc[0] for doc in list(topics_docs_dic.values())]

vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=1, max_features=None, stop_words=None, tokenizer=lambda x: x.split())
tfidf_matrix = vectorizer.fit_transform(topics_docs)
# print(vectorizer.get_feature_names_out())
# print(tfidf_matrix.tolist())

def document_likelihood(doc, test_doc_topics, topics_terms_list):
    log_likelihood = 0

    # checks if there was no prediction, likelihood will be 0
    if not test_doc_topics:
        return 0

    doc_length = len(doc)
    for word_id, word_count in doc:  # loop over words
        word_topic_probability = 0
        for topic_id, topic_prob in test_doc_topics:
            # gets words per topic from model
            word_prob = 1e-6
            for word_prob_pair in topics_terms_list[topic_id]:
                if word_id == word_prob_pair[0]:
                    word_prob = word_prob_pair[1]
                    break
            word_topic_probability += word_prob * topic_prob  # probability of word over all topics
        log_likelihood += word_count * np.log2(word_topic_probability)
    if log_likelihood == -np.inf: return 0
    return log_likelihood

def calculate_perplexity(test_doc_topics, topics_terms_list, test_corpus):
    # calculate perplexity
    docs_perplexity = []  # keeps perplexity for each doc

    for doc_index, doc in enumerate(test_corpus):
        doc_likelihood = document_likelihood(doc, test_doc_topics[doc_index], topics_terms_list)
        docs_perplexity.append(doc_likelihood)

    total_terms_in_test_docs = np.sum([count for doc in test_corpus for _, count in doc])
    perplexity_score = -1 * np.sum(docs_perplexity) / (10 * total_terms_in_test_docs)
    return perplexity_score

def calculate_coherence(topic_tokens_dic, num_topics, sliding_window):
    pmi = np.zeros(shape=num_topics)
    for topic in range(num_topics):
        topic_tokens_dist = topic_tokens_dic[topic]
        tokens_in_topic = [token_id for token_id, _ in topic_tokens_dist]
        prob_tokens_in_topic = [dictionary.dfs[token]/num_words for token in tokens_in_topic]
        word_pairs = [[tokens_in_topic[i], tokens_in_topic[j]] for i in range(len(tokens_in_topic)) for j in range(i+1, len(tokens_in_topic))]
        word_pairs_count = np.zeros(shape=len(word_pairs))
        for index, pair in enumerate(word_pairs):
            for doc in corpus:
                if pair[0] in dict(doc) and pair[1] in dict(doc):
                    word_pairs_count[index] += 1

        prob_token_pairs = word_pairs_count / math.comb(num_words, 2)

        total_co_occurrence = 0
        for index, pair in enumerate(word_pairs):
            total_co_occurrence += co_occurrence_calculation(prob_token_pairs, prob_tokens_in_topic, index, tokens_in_topic.index(pair[0]), tokens_in_topic.index(pair[1]))

        pmi[topic] = total_co_occurrence / math.comb(sliding_window, 2)
    return np.square(np.sum(np.power(1-pmi, 2)))

def co_occurrence_calculation(prob_token_pairs, prob_tokens_in_topic, index, token_1, token_2):
    if (prob_token_pairs[index] == 0):
        return -1
    return ((np.log(prob_tokens_in_topic[token_1]) + np.log(prob_tokens_in_topic[token_2])) / np.log(prob_token_pairs[index])) - 1

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
num_exploration = 20
num_iterations = 20
beta = 0.04  # initial percentage of taken from heuristic (global pheromone) - percentage changes through iterations
theta = 0.3  # effect of exploration ants pheromones (local)
evaporation_rate = 0.2
epsilon = 0.000001
lda_passes_count = 20
sliding_window = 10
load_exploration = True

save_prerequirments(np.array(sample_keys), np.array(train_doc_ids), np.array(test_doc_ids))

# save the best global solution (at each iteration)
iteration_start, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics = get_iteration_data()

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
        exploration_start, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics = get_exploration_data()
        if exploration_start >= num_exploration: exploration_start = 0
        load_exploration = False

    beta = beta + 0.05  # update pheromone effect percentage

    # generates the first group of ants here!
    if exploring_ants is None:
        exploring_ants = np.random.rand(num_ants, num_topics, num_words)

    for exploration_index in range(exploration_start, num_exploration):

        # keeps fitness function of generated ants
        exploring_ants_fitness_val = np.array([])

        # keeps exploring ants coverage, coherence, and preplexity
        exploring_ants_metrics = []

        exploring_topic_distribution = np.empty(shape=(num_ants, num_topics, num_words))

        # add the effect of global pheromone
        for ant_index in range(num_ants):
            exploring_topic_distribution[ant_index] = np.multiply(exploring_ants[ant_index] * (1-beta), pheromone_matrix * beta)  # pheromone and ant have different dimensions
            exploring_topic_distribution[ant_index] = preprocessing.normalize(exploring_topic_distribution[ant_index], axis=0, norm='l1')

        for topic_distribution in exploring_topic_distribution:
            # Evaluate the quality of the new topic distribution
            new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto',
                                      eta=topic_distribution, passes=lda_passes_count)

            coverages = ut.calculate_coverage(num_documents, num_topics, num_words, new_lda, corpus)
            coverage_score = np.power(np.sum(np.power(coverages, 2)) / num_documents, 0.5)
            # coherence_score = models.CoherenceModel(model=new_lda, texts=tokenized_documents_arr, dictionary=dictionary, coherence='c_uci').get_coherence() * -1

            topic_tokens_dic = {}
            for topic_id, _ in enumerate(favorite_topics):
                topic_tokens_dic[topic_id] = new_lda.get_topic_terms(topic_id, sliding_window)
            coherence_score = calculate_coherence(topic_tokens_dic, num_topics, sliding_window)

            test_doc_topics = {}  # keeps topics of each document
            test_topic_terms_dic = {}  # keeps the terms and their probabilities under each topic

            for doc_id, doc in enumerate(test_corpus):
                test_doc_topics[doc_id] = new_lda.get_document_topics(doc)

            for topic_id, _ in enumerate(favorite_topics):
                test_topic_terms_dic[topic_id] = new_lda.get_topic_terms(topic_id, num_words)

            topics_terms_list = list(test_topic_terms_dic.values())
            topics_terms_list = [[list(term_prob_pair) for term_prob_pair in term_prob_list] for topic_id, term_prob_list in enumerate(topics_terms_list)]
            perplexity_score = calculate_perplexity(test_doc_topics, topics_terms_list, test_corpus)

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
        selected_leader_ants = ut.roulette_wheel(exploring_ants_fitness_val)

        # generate new ants
        exploring_ants = np.random.rand(num_ants, num_topics, num_words)

        # include local pheromone in new ants path
        for ant_index in range(num_ants):
            exploring_ants[ant_index] = np.multiply(exploring_ants[ant_index] * (1-theta), exploring_topic_distribution[selected_leader_ants[ant_index]] * theta)
            exploring_ants[ant_index] = preprocessing.normalize(exploring_ants[ant_index], axis=0, norm='l1')

        save_exploration_data(exploration_index+1, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics)

    if exploration_best_solution_score < best_solution_score:
        best_iteration = iteration
        best_metrics = exploration_best_metrics
        best_solution = exploration_best_solution
        best_solution_score = exploration_best_solution_score

    # Update the pheromone matrix
    pheromone_matrix = np.add(pheromone_matrix, exploration_best_solution)
    pheromone_matrix *= (1 - evaporation_rate)
    pheromone_matrix = preprocessing.normalize(pheromone_matrix, axis=0, norm='l1')

    save_iteration_data(iteration+1, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics)

    print("\n======================")
    print("Current iteration: ", iteration,
          "\nObjective function: ", exploration_best_solution_score,
          "\nBest Distribution: \n", exploration_best_solution,
          "\nCoverage", exploration_best_metrics[0],
          "\nCoherence: ", exploration_best_metrics[1],
          "\nPreplexity: ", exploration_best_metrics[2])

    # Print the best solution found so far
    print("\n======================")
    print('Best iteration: ', best_iteration, '\nBest Metrics: ', best_metrics, '\nBest objective function: ', best_solution_score, '\nBest solution: \n', best_solution)

    print('Iteration', iteration, ' duration: ', time.time() - start_time)
    print("\n====================================================================================")

########################################################################################################################
# test Section
new_lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto', eta=best_solution, passes=50)
predicted_topics = []
for index in range(len(test_corpus)):
    predictions = new_lda.get_document_topics(test_corpus[index])

    # Get the topic IDs
    topics = [p[0] for p in predictions]

    # Get the probabilities
    probs = [p[1] for p in predictions]

    # Find index of max probability
    max_index = probs.index(max(probs))

    # The topic with highest probability
    predicted_topics.append(topics[max_index])

from sklearn.metrics import f1_score

true_topics = [topic_list[0] for topic_list in test_topics_ids]

f1 = f1_score(true_topics, predicted_topics, average='macro')

print(f1)

f1_scores = {}
for t in range(num_topics):
   f1 = f1_score(true_topics, predicted_topics, average=None)[t]
   f1_scores[t] = f1

print(f1_scores)

from sklearn.metrics import classification_report
print(classification_report(true_topics, predicted_topics))

lengths = []
sum = [0,0,0,0]
count = [0,0,0,0]

for index, doc in enumerate(tokenized_documents_arr):
  topic = train_topics_ids[index][0]
  sum[topic] = sum[topic] + len(doc)
  count[topic] = count[topic] + 1

for index, value in enumerate(sum):
    print("topic {}:\tCount: {},\tAvg length: {}\n".format(index, count[index], sum[index]/count[index]))