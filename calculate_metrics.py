import numpy as np
import os
import pickle
import pandas as pd
import argparse


# Define thresholds for preprocessing
dgp = 1
thresholds = np.arange(0.0, 0.1, 0.005)
k_true = 50

corpus_path = 'results/corpora'
model_path = 'results/models'

average_distances_all = []
coherence_all = []
topic_similarity_all = []
perplexity_all = []

# from tmtoolkit
def _get_normalized_topic_word_distrib(lda_instance):
    return lda_instance.components_ / lda_instance.components_.sum(axis=1)[:, np.newaxis]
from tmtoolkit.topicmod.evaluate import metric_cao_juan_2009, metric_coherence_mimno_2011
from tm_functions import *

# identify the current threshold considered
t_ind = 0
t = round(thresholds[t_ind], 3)
# corpus ID
_id = 1
average_distances = []
coherence = []
topic_similarity = []
perplexity = []

lda = pickle.load(open(f'{model_path}/corpus_{_id}/lda_mindf_{t}', 'rb'))
word_topic_true = pd.read_pickle(open(f'{corpus_path}/true_word_topic_{_id}', 'rb'))
count_data = pickle.load(open(f"{corpus_path}/count_data_doc_freq_{_id}_{t}", 'rb'))
vocabulary = pickle.load( open(f"{corpus_path}/vocab_doc_freq_{_id}_{t}", 'rb'))

## First step
tw = topic_word_df_sklearn(lda, vocabulary)
best_matching, match, row_ind, col_ind = match_hungarian(word_topic_true, tw, method='cosine', merge = 'outer')
# retrieve best distances (as identified in one-to-one matching)
best_distances = [match.iloc[i, j] for i,j in zip(row_ind, col_ind)]
average_distance = np.mean(best_distances)

## Second step
# Use tmtoolkit functions to calculate coherence, topic similarity etc.
topic_word_distrib = _get_normalized_topic_word_distrib(lda)
perplexity = lda.perplexity(count_data)
coherence = metric_coherence_mimno_2011(topic_word_distrib,
                                        count_data, return_mean=True)
topic_similarity = metric_cao_juan_2009(topic_word_distrib)


### COSINE SIMILARITY
# do best matching in the cases the topic was assigned multiple times,
# keep the match with the highest cosine similarity value
assignment, values = topic_word_frequencies_matching(word_topic_true,
                                                     tw,
                                                     measure='cosine similarity',
                                                     vocabulary='union')
# construct a data frame that contains a matched topic number as well as cosine similarity value
df_matches = pd.DataFrame({'topic_true': range(word_topic_true.shape[1]),
                           'topic_estimated': assignment,
                           'cosine_similarity': values})

# cut off value
cut_off = 0.8
matches_filtered = df_matches[df_matches.cosine_similarity >= cut_off].sort_values(
    'cosine_similarity').drop_duplicates('topic_estimated', keep='last').sort_index()
relevant_items_retrieved = matches_filtered.shape[0]
relevant_items = k_true
recall_cosine = relevant_items_retrieved / relevant_items


### JENSEN-SHANNON
# do best matching in the cases the topic was assigned multiple times,
# keep the match with the lowest Jensen-Shannon value
assignment, values = topic_word_frequencies_matching(word_topic_true,
                                                     tw,
                                                     measure='js distance',
                                                     vocabulary='union')
# construct a data frame that contains a matched topic number
# as well as cosine similarity value
df_matches = pd.DataFrame({'topic_true': range(word_topic_true.shape[1]),
                           'topic_estimated': assignment,
                           'JS': values})

# cut off value
cut_off_js = 0.2

matches_filtered = df_matches[df_matches.JS <= cut_off_js].sort_values('JS').drop_duplicates('topic_estimated',
                                                                                          keep='first').sort_index()
relevant_items_retrieved = matches_filtered.shape[0]
relevant_items = k_true
recall_jensen = relevant_items_retrieved / relevant_items


### Ranked Biased Overlap
n_top_words = 100
topics_estimated = get_topic_words_sklearn_tf(lda, vocabulary, n_top_words, frequency=False)
all_topics = []
words = word_topic_true.index.values
for topic_idx, topic in enumerate(word_topic_true.T.to_numpy()):
    topic_words = " ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
    all_topics.append(topic_words)
# do best matching: in the cases the topic was assigned multiple times,
# keep the match with the highest cosine similarity value
assignment, values = rbo_based_matching(all_topics, topics_estimated, 0.95)
# construct a data frame that contains a matched topic number as well as cosine similarity value
df_matches = pd.DataFrame({'topic_true': range(word_topic_true.shape[1]),
                           'topic_estimated': assignment,
                           'rbo': values})

# cut off value
cut_off_rbo = 0.8
matches_filtered = df_matches[df_matches.rbo >= cut_off_rbo].sort_values('rbo').drop_duplicates('topic_estimated',
                                                                                            keep='last').sort_index()
relevant_items_retrieved = matches_filtered.shape[0]
relevant_items = k_true
recall_rbo = relevant_items_retrieved / relevant_items

np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_ave_dist_{t}', average_distance)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_coherence_{t}', coherence)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_cao_juan_{t}', topic_similarity)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_perplexity_{t}', perplexity)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_cosine_{round(cut_off, 2)}_{t}', recall_cosine)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_jensen_{round(cut_off_js, 2)}_{t}', recall_jensen)
np.save(f'results/topic_quality_metrics/dgp{dgp}_{_id}_rbo_{round(cut_off_rbo, 2)}_{t}', recall_rbo)


