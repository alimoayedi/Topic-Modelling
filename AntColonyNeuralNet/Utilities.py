import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def file_exists(directory):
    """Check if a file exists in the current directory"""
    return os.path.exists(directory) and os.path.isfile(directory)


def set_requirements(sample_key, train_doc_ids, test_doc_ids):
    directory = 'outputs/pre_requirements.npz'
    np.savez(directory, sample_key=sample_key, train_doc_ids=train_doc_ids, test_doc_ids=test_doc_ids)


def get_requirements():
    if not file_exists(r'/outputs/pre_requirements.npz'):
        return None, None, None
    data = np.load(r'outputs/pre_requirements.npz')
    sample_key = data['sample_key']
    train_doc_ids = data['train_doc_ids']
    test_doc_ids = data['test_doc_ids']
    return sample_key, train_doc_ids, test_doc_ids


def save_exploration_data(next_exploration_index, exploring_ants, exploration_best_solution,
                          exploration_best_solution_score, exploration_best_metrics):
    directory = 'outputs/exploration_data.npz'
    np.savez(directory,
             next_exploration_index=next_exploration_index,
             exploring_ants=exploring_ants,
             exploration_best_solution=exploration_best_solution,
             exploration_best_solution_score=exploration_best_solution_score,
             exploration_best_metrics=exploration_best_metrics)


def get_exploration_data():
    if not file_exists(r'/outputs/exploration_data.npz'):
        return 0, None, None, float('inf'), None

    data = np.load(r'outputs/exploration_data.npz')
    next_exploration_index = int(data['next_exploration_index'])
    exploring_ants = data['exploring_ants']
    exploration_best_solution = data['exploration_best_solution']
    exploration_best_solution_score = int(data['exploration_best_solution_score'])
    exploration_best_metrics = data['exploration_best_metrics']

    return next_exploration_index, exploring_ants, exploration_best_solution, exploration_best_solution_score, exploration_best_metrics


def save_iteration_data(next_iteration_index, pheromone_matrix, best_iteration, best_solution, best_solution_score,
                        best_metrics):
    directory = 'outputs/iteration_data.npz'
    np.savez(directory,
             Next_iteration_index=next_iteration_index,
             pheromone_matrix=pheromone_matrix,
             best_iteration=best_iteration,
             best_solution=best_solution,
             best_solution_score=best_solution_score,
             best_metrics=best_metrics)


def get_iteration_data():
    if not file_exists(r'/outputs/iteration_data.npz'):
        return 0, None, 0, None, float('inf'), None

    data = np.load(r'outputs/iteration_data.npz')
    next_iteration_index = int(data['Next_iteration_index'])
    pheromone_matrix = data['pheromone_matrix']
    best_iteration = int(data['best_iteration'])
    best_solution = data['best_solution']
    best_solution_score = int(data['best_solution_score'])
    best_metrics = data['best_metrics']
    return next_iteration_index, pheromone_matrix, best_iteration, best_solution, best_solution_score, best_metrics


def get_cosine_similarity_score(tokenized_documents):
    joined_docs = [" ".join(words) for words in tokenized_documents.values()]

    vectorized = TfidfVectorizer()
    vectorized_docs = vectorized.fit_transform(joined_docs)
    vectorized_docs = vectorized_docs.transpose()
    return cosine_similarity(vectorized_docs)


def filter_documents(tokenized_documents, cosine_sim_score):
    filtered_tokenized_documents = []

    for doc in tokenized_documents.values():
        to_remove = set()
        for item_1 in range(len(doc)):
            for item_2 in range(item_1 + 1, len(doc)):
                if cosine_sim_score[item_1, item_2] > 0.9:
                    to_remove.add(doc[item_1])
        filtered_tokenized_documents.append([term for term in doc if term not in to_remove])


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


def get_number_words_in_topic(topic_term_probability):
    count = sum(1 for pair in topic_term_probability if pair[1] > 9e-5)
    return count


def calculate_coverage(num_documents, num_topics, num_words, model, corpus):
    n_tk = np.zeros(shape=num_topics)
    proportions = np.zeros(shape=num_topics)
    coverages = np.zeros(shape=num_documents)

    # finds the number of terms each topic has.
    for topic in range(num_topics):
        n_tk[topic] = get_number_words_in_topic(model.get_topic_terms(topic, num_words))

    # finds the total number of terms in each document
    total_tf = [sum(freq for _, freq in corpus[doc]) for doc in range(num_documents)]

    for doc in range(num_documents):
        doc_topics, word_topics, _ = model.get_document_topics(bow=corpus[doc], per_word_topics=True)
        doc_topics = [topic_id for topic_id, _ in doc_topics]
        for topic in doc_topics:
            sum_words_topic_tf = 0
            document_topic_num_words = 0
            for word_id, word_topic_list in word_topics:
                if topic in word_topic_list:
                    document_topic_num_words += 1  # number of terms with specific topic in each document
                    sum_words_topic_tf += [item[1] for item in corpus[doc] if item[0] == word_id][0]
            if document_topic_num_words > n_tk[topic]:
                n_tk[topic] = document_topic_num_words
                print("**** An error in number of topic terms occurred. ****")
            proportions[topic] = sum_words_topic_tf / ((n_tk[topic] - document_topic_num_words) + 1)


        under_root = 0
        for word_id, word_tf in corpus[doc]:
            p1 = word_tf / total_tf[doc]
            p2 = 0
            for topic in doc_topics:
                p2 += (word_tf / sum_words_topic_tf) * proportions[topic] if sum_words_topic_tf != 0 else 0
            under_root += np.power(p1 - p2, 2)
        coverages[doc] = np.power(under_root, 0.5)
    return coverages


def roulette_wheel(ants_fitness):
    num_ants = len(ants_fitness)
    fitness_ranks = np.unique(ants_fitness, return_inverse=True)[1] + 1
    scaled_fitness = np.divide(ants_fitness, fitness_ranks)
    sum_scaled_fitness = np.sum(scaled_fitness)
    selection_probs = np.divide(scaled_fitness, sum_scaled_fitness)
    return np.random.choice(num_ants, p=selection_probs, size=num_ants)








