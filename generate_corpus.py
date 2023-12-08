import numpy as np
import pandas as pd
import os
from numpy.random import default_rng
from scipy.stats import poisson
import time
import pickle
import argparse


def generate_text_from_lda(text_length, doc_topic_weights, topic_word_weights, vocabulary, rng):
    '''
    :param corpus_length: a list containing the length of texts in the corpus.
    :param doc_topic_weights: a data frame, where the rows correspond to the documents and the columns - to the topics
    :param topic_word_weights: a data frame, where the rows correspond to the topics and the columns - to the vocabulary words
    :param vocabulary: list of vocabulary words
    :param rng: a random number generator object to use for this function.
    :return: generated corpus of texts
    '''

    import numpy as np
    from tqdm import tqdm
    #np.random.seed(random_seed)
    corpus_gen = []

    word_weights_cum = np.cumsum(topic_word_weights, axis=1)
    topic_weights_cum = np.cumsum(doc_topic_weights, axis=1)

    for j, text in enumerate(text_length):
        # define the length of a new abstract
        abstract_length = text
        # cumulative sum of topic probabilities of the abstract
        topic_prob_cum = topic_weights_cum.iloc[j]
        # generate uniformly distributed floats for topic and words selection
        deltas = rng.uniform(0, 1, abstract_length)
        delta2s = rng.uniform(0, 1, abstract_length)

        abstract_gen = []

        for k in range(0, abstract_length):
            # select randomly a topic
            ii, = np.where(topic_prob_cum >= deltas[k])
            topic = min(ii)
            # select randomly a word
            ii, = np.where(word_weights_cum.iloc[topic] >= delta2s[k])
            new_word_index = min(ii)
            abstract_gen.append(vocabulary[new_word_index])
        corpus_gen.append(' '.join(abstract_gen))
    return corpus_gen


start = time.time()

# Random Generator
rng = default_rng(seed=123)
# Corpus ID
_id = 1
# Path for results storage
path = 'results/corpora'
# Define the properties of the data
n_words = 30000  # number of unique terms
n_topics = 50  # number of topics
text_length = 3000  # the average text length
corpus_size = 1000  # number of document in the corpus
docs_length = poisson.rvs(mu=text_length, size=corpus_size, random_state=rng)

# Using symmetric (flat) priors
# Choose a prior for document-topic distribution (alpha)
alpha = tuple([1/n_topics]*n_topics)  # n_topics, a distribution over topics is drawn for each document
topic_doc = pd.DataFrame(rng.dirichlet(alpha, corpus_size))
topic_doc.index = [f'doc{i}' for i in range(1, corpus_size+1, 1)]
topic_doc = topic_doc.T

# Choose a prior for word-topic distribution
beta = tuple([1/n_topics]*n_words)  # n_words, a distribution over word sis drawn for each topic
word_topic = pd.DataFrame(rng.dirichlet(beta, n_topics)).T
word_topic.index = [f'word{i}' for i in range(1, n_words+1, 1)]
dictionary = word_topic.index

pickle.dump(topic_doc, open(f"{path}/true_topic_doc_{_id}", 'wb'))
pickle.dump(word_topic, open(f"{path}/true_word_topic_{_id}", 'wb'))

# Generate corpora using the true document-topic and topic-word distributions

corpus = generate_text_from_lda(docs_length, topic_doc.T, word_topic.T, dictionary, rng)
pd.Series(corpus).to_pickle(f"{path}/corpus_{_id}.pkl")
print(f"Generating corpus took {time.time()-start}")

