import numpy as np
import tensorflow as tf
from nltk.corpus import reuters
import Preprocess
from gensim import corpora, models


# raw data and categories
documents = reuters.fileids()
train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs = list(filter(lambda doc: doc.startswith("test"), documents))
categories = reuters.categories()

# Preprocess data
documents_topics, tokenized_documents, unique_tokens = Preprocess.preprocess(categories, train_docs)

# Create a dictionary of unique words
# dictionary = corpora.Dictionary(tokenized_documents)

# Convert the corpus into a bag-of-words representation
# corpus = [dictionary.doc2bow(document) for document in tokenized_documents]


# Load your preprocessed data and tokens
data = ...  # Your preprocessed data (tokenized documents)
tokens = ...  # Tokens for each document
topics = ...  # Related topic of each document


# Set the random seed for reproducibility
tf.random.set_seed(42)

# Define the number of topics in your dataset
num_topics = len(categories)
num_documents = len(tokenized_documents)
num_words = len(unique_tokens)

# Convert tokens and topics into numerical representations
vocab = set([token for doc in tokenized_documents for token in doc])
vocab_size = len(vocab)

# Create a mapping between tokens and indices
token2idx = {token: idx for idx, token in enumerate(vocab)}
topic2idx = {topic: idx for idx, topic in enumerate(set(topics))}

# Convert tokens and topics to numerical indices
token_indices = [[token2idx[token] for token in doc_tokens] for doc_tokens in tokens]
topic_indices = [topic2idx[topic] for topic in topics]

# Pad sequences to ensure equal length for training
max_seq_length = max(len(indices) for indices in token_indices)
token_indices_padded = tf.keras.preprocessing.sequence.pad_sequences(token_indices, maxlen=max_seq_length)

# Split the data into training and validation sets
train_ratio = 0.8
train_size = int(len(data) * train_ratio)
train_data = token_indices_padded[:train_size]
train_topics = topic_indices[:train_size]
val_data = token_indices_padded[train_size:]
val_topics = topic_indices[train_size:]

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_seq_length),
    tf.keras.layers.GRU(256),
    tf.keras.layers.Dense(num_topics, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
num_epochs = 10
model.fit(train_data, train_topics, validation_data=(val_data, val_topics), batch_size=batch_size, epochs=num_epochs)

# Once the model is trained, you can use it for topic prediction on new documents
new_document = ...  # Your new document to predict the topic for
new_tokens = ...  # Tokenize the new document
new_token_indices = [token2idx[token] for token in new_tokens]
new_token_indices_padded = tf.keras.preprocessing.sequence.pad_sequences([new_token_indices], maxlen=max_seq_length)

# Predict the topic for the new document
predicted_topic = model.predict(new_token_indices_padded)
predicted_topic = np.argmax(predicted_topic)

# Map the predicted index back to the original topic label
predicted_topic_label = list(topic2idx.keys())[list(topic2idx.values()).index(predicted_topic)]

# Print the predicted topic label
print("Predicted topic:", predicted_topic_label)
