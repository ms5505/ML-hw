#Python script for unigrams and tf-idf representations

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Preprocessed document with stopwords removed
train_preprocessed = pd.read_csv('train_preprocessed2.csv', sep = ',')

X_train_pp = train_preprocessed["text"]
y_train_pp = train_preprocessed['label']
y_train_pp.as_matrix()   # convert pandas series to array
y_train_pp = y_train_pp.values.reshape([y_train_pp.shape[0], 1])  # reshape to fit into sparse matrix later

# Create dictionary that contains unique words as key and number of documents in D that contains word as value
for index, text in X_train_pp.iteritems():
    if type(text) == float: continue
    else: words = text.split(' ')
    words = [word for word in words if '_' not in word]    #further preprocessing
    unique_words = set(words)
    for i in unique_words:
        if i not in docs_count: docs_count[i] = 1
        else: docs_count[i] += 1

# remove words that only appear in one document out of D
from collections import OrderedDict
docs_count = {k:v for k,v in docs_count.items() if v != 1}
unigrams_docs_count = OrderedDict(sorted(docs_count.items()))


# Map words in unigram words to integer for sparse matrix construction
unigram_map = {}
for x in enumerate(unigrams_docs_count):
  val, key = x
  unigram_map[key] = val

def get_key(index):
    return list(unigram_map.keys())[list(unigram_map.values()).index(index)]


# Build sparse matrix for unigrams
def build_sparse_unigram(X_data, unigram_map):
    # initialize sparse matrix
    dok = sp.dok_matrix((10 ** 6, len(unigram_map) + 2), dtype=np.float64)

    def get_word_index(word):
        return unigram_map[word]

    for index, text in X_data.iteritems():
        wordfreq = {}
        if type(text) == float:
            continue
        else:
            words = text.split(' ')
        words = [word for word in words if word in unigram_map]
        # create dictionary for word frequency in d
        for word in words:
            if word not in wordfreq:
                wordfreq[word] = 1
            else:
                wordfreq[word] += 1

        for k, v in wordfreq.items():
            dok[index, get_word_index(k)] = v

    return dok

unigram_dok = build_sparse_unigram(X_train_pp, unigram_map)

# Add w_0 and y_train vectors to sparse unigram matrix
w_0 = np.ones([1000000, 1])
unigram_dok[:, len(unigram_map)] = w_0
unigram_dok[:, len(unigram_map)+1] = y_train_pp

# convert unigram dok_matrix to csr_matrix for faster row-wise operations
unigram_csr = unigram_dok.tocsr()

# Build sparse matrix for tf-idf
def build_sparse_tfidf(X_data, unigram_map, unigrams_docs_count):
    # initialize sparse matrix
    dok = sp.dok_matrix((X_data.shape[0], len(unigram_map) + 2), dtype=np.float64)
    D = X_train_pp.shape[0]

    def get_word_index(word):
        return unigram_map[word]

    def idf(t):
        return D / unigrams_docs_count.get(t)

    def tf_idf(t, tf):
        return tf * np.log10(idf(t))

    for index, text in X_data.iteritems():
        wordfreq = {}
        if type(text) == float:
            continue
        else:
            words = text.split(' ')
        words = [word for word in words if word in unigram_map]
        for word in words:
            if word not in wordfreq:
                wordfreq[word] = 1
            else:
                wordfreq[word] += 1

        for k, v in wordfreq.items():
            tfidf = tf_idf(k, v)
            dok[index, get_word_index(k)] = tfidf

    return dok

# Construct tf-idf sparse matrix
tfidf_dok = build_sparse_tfidf(X_train_pp, unigram_map, unigrams_docs_count)
w_0 = np.ones([1000000, 1])
tfidf_dok[:, len(unigram_map)] = w_0
tfidf_dok[:, len(unigram_map)+1] = y_train_pp
tfidf_csr = tfidf_dok.tocsr() # Convert to csr_matrix for faster row-wise operations


def train(train_csr, word_index_map, n_epoch=2, sample_frac=0.8):
    '''
    Return: a tuple of number of errors made in each epoch, and the final averaged weight vector.
    '''
    # Initialize w vector of zeros
    w = np.zeros([1, train_csr.shape[1] - 1])
    w_sum = 0

    # Compute dot product of w and x vectors
    def net_input(x):
        return (np.dot(w, x.T))

    # Compute predicted y
    def predict(x):
        return (np.where(net_input(x) >= 0.0, 1, 0))

    train_accuracy = []
    for epoch in range(n_epoch):
        n = 0
        error = 0

        # Get training samples and shuffle the data
        train_size = int(sample_frac * train_csr.shape[0])
        index = np.arange(train_csr.shape[0])
        np.random.shuffle(index)
        if sample_frac != 1.0:
            train_index = index[:train_size]
            train_set = train_csr[train_index, :]
            test_set = train_csr[-train_index, :]
        else:
            train_set = train_csr

        for i in train_set:
            d = train_set[i].toarray()
            y = d[:, -1:]  # label
            x = d[:, :-1]  # x array
            update = y - predict(x)
            w += update * x
            error += int(update != 0.0)
            n += 1
            if epoch == 1:
                w_sum += w
        train_accuracy.append((train_set.shape[0] - error) / train_set.shape[0])

    w_final = w_sum / n + 1

    ## Test set
    # Compute dot product of w_final and x vectors
    if sample_frac != 1.0:
        def net_input_test(x):
            return (np.dot(w_final, x.T))

        # Compute predicted y
        def predict_test(x):
            return (np.where(net_input_test(x) >= 0.0, 1, 0))

        test_error = 0
        for i in test_set:
            d = i.toarray()
            y = d[:, -1:]  # label
            x = d[:, :-1]  # x array
            if predict_test(x) != y: test_error += 1
        test_accuracy = (test_set.shape[0] - test_error) / test_set.shape[0]

    else:
        test_accuracy = 0

    return (train_accuracy, test_accuracy, w_final)


unigram_train_results = train(unigram_csr, unigram_map, sample_frac = .9)
tfidf_train_results = train(tfidf_csr, unigram_map, sample_frac = .9)


# Import test set
test_data = pd.read_csv('reviews_te.csv', sep = ',')
X_test = test_data["text"]
y_test = test_data['label']
y_test.as_matrix()
y_test = y_test.values.reshape([y_test.shape[0], 1])

# construct unigram test matrix
test_unigram_dok = build_sparse_unigram(X_test, unigram_map)
w_0 = np.ones([X_test.shape[0], 1])
test_unigram_dok[:, len(unigram_map)] = w_0
test_unigram_dok[:, len(unigram_map)+1] = y_test
test_unigram_csr = test_unigram_dok.tocsr()



# construct tf-idf test matrix
test_tfidf_dok = build_sparse_tfidf(X_test, unigram_map, unigrams_docs_count)
w_0 = np.ones([X_test.shape[0], 1])
test_tfidf_dok[:, len(unigram_map)] = w_0
test_tfidf_dok[:, len(unigram_map)+1] = y_test
test_tfidf_csr = test_tfidf_dok.tocsr()


# Test function
def test_prediction(test_csr, training_results):
    # Get final weights from training set
    w = training_results[2]

    # Compute dot product of w and x vectors
    def net_input(x):
        return (np.dot(w, x.T))

    # Compute predicted y
    def predict(x):
        return (np.where(net_input(x) >= 0.0, 1, 0))

    test_error = 0
    for i in test_csr:
        d = i.toarray()
        y = d[:, -1:]  # label
        x = d[:, :-1]  # x array
        if predict(x) != y: test_error += 1

    return (test_csr.shape[0] - test_error) / test_csr.shape[0]


unigram_test_result = test_prediction(test_tfidf_csr, unigram_train_results)
tfidf_test_result = test_prediction(test_tfidf_csr, tfidf_train_results)

