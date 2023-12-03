import numpy as np

def lda(documents, num_topics, num_iter=100, alpha=0.1, beta=0.01):
    # Step 1: Tokenize the documents and create vocabulary
    vocabulary = set(word for document in documents for word in document)
    word_to_id = {word: idx for idx, word in enumerate(vocabulary)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    num_documents = len(documents)
    num_words = len(vocabulary)

    # Step 2: Initialize matrices
    theta = np.random.dirichlet([alpha] * num_topics, num_documents)
    phi = np.random.dirichlet([beta] * num_words, num_topics)

    # Step 3: EM Algorithm
    for _ in range(num_iter):
        # E-step: Update document-topic assignments
        for d, document in enumerate(documents):
            for n, word in enumerate(document):
                word_id = word_to_id[word]
                p_topic = theta[d] * phi[:, word_id]
                p_topic /= p_topic.sum()
                z = np.random.choice(num_topics, p=p_topic)
                theta[d, z] += 1
                phi[z, word_id] += 1

        # M-step: Update topic-word distributions
        theta /= theta.sum(axis=1, keepdims=True)
        phi /= phi.sum(axis=1, keepdims=True)

    return theta, phi, word_to_id, id_to_word

# Sample documents
documents = [
    ["machine", "learning", "is", "fascinating"],
    ["natural", "language", "processing", "is", "a", "subfield", "of", "ai"],
    ["python", "is", "a", "popular", "programming", "language", "for", "data", "science"],
    ["deep", "learning", "requires", "a", "lot", "of", "data"],
    ["topic", "modeling", "is", "an", "interesting", "field", "in", "natural", "language", "processing"]
]

# Number of topics
num_topics = 2

# Number of iterations
num_iter = 100

# Run LDA
theta, phi, word_to_id, id_to_word = lda(documents, num_topics, num_iter)

# Print the results
print("Document-Topic Distribution (Theta):")
print(theta)
print("\nTopic-Word Distribution (Phi):")
print(phi)

# Optional: Print the most probable words for each topic
for topic_id in range(num_topics):
    top_words_idx = np.argsort(phi[topic_id])[::-1][:5]
    top_words = [id_to_word[idx] for idx in top_words_idx]
    print(f"\nTop words for Topic {topic_id + 1}: {top_words}")
