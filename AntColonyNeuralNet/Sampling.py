import random
import math
import re
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# defined stop words
cachedStopWords = stopwords.words("english")


def get_sample(data_list, percentage):
    # Calculate the total length of arrays tokenized_documents
    total_length = len(data_list)

    # Determine the number of items to select based on percentage
    num_documents_to_select = int(total_length * percentage)

    # Shuffle the indices randomly
    random.shuffle(data_list)

    # Select the first num_items_to_select indices
    sample_data = data_list[:num_documents_to_select]

    return sample_data


def get_sample_equal_size(percentage, labels, dataset):
    # determines the total number of samples required from all labels
    num_samples = math.ceil(len(dataset) * percentage)
    num_left_labels = len(labels)

    # saves taken samples
    samples = []

    # get the list of documents under each label
    doc_labels_dic = {}
    for label in labels:
        docs_under_label = [doc_id for doc_id, doc_labels in dataset.items() if label in doc_labels]
        doc_labels_dic[label] = docs_under_label

    # Get the key and length as tuples
    num_docs_per_label = [(label, len(doc_list)) for label, doc_list in doc_labels_dic.items()]

    # Sort by length
    num_docs_per_label_sorted = sorted(num_docs_per_label, key=lambda x: x[1])

    # Get just the keys in order
    ordered_labels_by_length = [label for label, _ in num_docs_per_label_sorted]

    for label in ordered_labels_by_length:

        # checks if total number of samples is a multiplier of the number of lables
        # if not, from each label one extra sample is taken to fill the deficincy
        # complement_samples determines the number of extra samples required
        complement_samples = num_samples % num_left_labels

        # determines the number of required samples from each label
        if complement_samples != 0:
            # one extra sample is taken till the complement_samples == 0
            num_label_samples = num_samples // num_left_labels + 1
            complement_samples = complement_samples - 1
        else:
            num_label_samples = num_samples // num_left_labels

        # extracts all the entries with the specific label
        label_items = [doc_id for doc_id, doc_labels in dataset.items() if label in doc_labels]

        # if the number of samples if more than the entries with that label,
        # all the items are taken. otherwise a random sample is selected.
        if num_label_samples > len(label_items):
            label_samples = label_items
        else:
            label_samples = random.sample(label_items, num_label_samples)

        # save the taken sample
        samples.extend(label_samples)

        # subtract the number of taken samples from all the required samples
        num_samples = num_samples - len(label_samples)

        # subtract the number of labels by 1
        num_left_labels = num_left_labels - 1

    random.shuffle(samples)
    return samples


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def tokenize(text):
    min_length = 0
    lemmatizer = WordNetLemmatizer()
    p = re.compile('[a-zA-Z]+')
    words = map(lambda word: word.lower(), word_tokenize(text))
    filtered_tokens = [word for word in words if p.match(word) and len(word) > min_length]
    tokens = (token for token in filtered_tokens if token not in cachedStopWords)
    # stemmed_tokens = (PorterStemmer().stem(token) for token in filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return lemmatized_tokens


def preprocess_manual(train_docs):
    tokenized_documents = {}

    for doc_id, document in train_docs.items():
        tokenized_doc = tokenize(document)
        if len(tokenized_doc) > 2:
            tokenized_documents[doc_id] = tokenized_doc

    return tokenized_documents


def get_train_test_data(data_list, percentage):
    # Calculate the total length of arrays tokenized_documents
    total_length = len(data_list)

    # Determine the number of items to select (80% of total length)
    num_documents_to_select = int(total_length * percentage)

    # Shuffle the indices randomly
    random.shuffle(data_list)

    # Select the first num_items_to_select indices
    train_data = data_list[:num_documents_to_select]
    test_data = data_list[num_documents_to_select:]

    return train_data, test_data

