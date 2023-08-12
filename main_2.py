xs = [[0, 3, 4, 1, 2, 0, 3, 0],  # topics
      [0, 2, 1, 3, 1, 4, 2, 2],
      [2, 1, 1, 3, 2, 2, 2, 1],
      [1, 2, 3, 1, 2, 1, 1, 3]]


doc_term = [[5, 0, 1, 0, 0, 0, 0, 3],  # frequency
            [0, 1, 0, 0, 0, 1, 0, 5]]

term_doc_topic = np.zeros(shape=(5, 8, 4))

for doc_index, doc in enumerate(doc_term):
    term_topic = np.zeros(shape=(8, 5))
    for term_index, count in enumerate(doc):
        if count != 0:
            topic_index = xs[doc_index][term_index]
            term_topic[term_index, topic_index] += count
    term_doc_topic[:, :, doc_index] = np.array(term_topic).T.tolist()


from sklearn import preprocessing
import numpy as np

a = [[[0.3, 0.4, 0.5], [0.2, 0.5, 0.9]], [[0.1, 0.2, 0.2], [0.1, 0.1, 0.5]]]
a = np.array(a)
norm_arr = np.apply_along_axis(preprocessing.normalize, 0, a)

b = a.reshape(-1, a.shape[-1])

# Normalize each layer
norm_arr_2d = preprocessing.normalize(b, axis=0, norm='l1')

# Reshape back to original shape
norm_arr = norm_arr_2d.reshape(a.shape)

normalized_array = lambda arr: preprocessing.normalize(a[0], axis=0, norm='l1')

norm_arr = np.apply_along_axis(normalized_array, 0, a)
a[0:1]