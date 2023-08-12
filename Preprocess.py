import multiprocessing
import Tokenization
from nltk.corpus import reuters


def tokenize_parallel(doc):
    return Tokenization.tokenize(reuters.raw(doc))


def tokenize_parallel_manual(doc_ids, train_docs):
    return Tokenization.tokenize(train_docs[doc_ids])

# def preprocess(categories, train_docs):
#     tokenized_documents = []
#     documents_topics = []
#     for doc in train_docs:
#         # tokenize the document
#         tokenized_doc = Tokenization.tokenize(reuters.raw(doc))
#         if len(tokenized_doc) != 0:
#             tokenized_documents.append(tokenized_doc)
#         documents_topics.append([categories.index(topic) for topic in reuters.categories(doc)])
#     unique_tokens = list(set(term for doc in tokenized_documents for term in doc))
#     return documents_topics, tokenized_documents, unique_tokens


def preprocess_routers(categories, train_docs, minimum_doc_length=0):

    tokenized_documents = []
    documents_topics = []

    with multiprocessing.Pool() as pool:
        tokenized_documents_pool = pool.map(tokenize_parallel, train_docs)
        for doc_index, doc in enumerate(tokenized_documents_pool):
            if len(doc) != 0:
                tokenized_documents.append(doc)
                documents_topics.append(
                    [categories.index(topic) for topic in reuters.categories(train_docs[doc_index])])

    unique_tokens = list(set(term for doc in tokenized_documents for term in doc))

    return documents_topics, tokenized_documents, unique_tokens


def preprocess_manual(train_docs):

    tokenized_documents = []

    with multiprocessing.Pool() as pool:
        tokenized_documents_pool = pool.map(tokenize_parallel_manual, train_docs.keys(), train_docs)
        for doc in enumerate(tokenized_documents_pool):
            if len(doc) != 0:
                tokenized_documents.append(doc)

    unique_tokens = list(set(term for doc in tokenized_documents for term in doc))

    return tokenized_documents, unique_tokens

