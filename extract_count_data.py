import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define thresholds for preprocessing
thresholds = np.arange(0.0, 0.1, 0.005)
k_true = 50
# Corpus ID
_id = 1
# corpus path
path = 'results/corpora'

corpus = pd.read_pickle(open(f"{path}/corpus_{_id}.pkl", 'rb'))
# count data with absolute term frequencies 
count_vectorizer_initial = CountVectorizer()
count_data_initial = count_vectorizer_initial.fit_transform(corpus)
# initial vocabulary
vocab_df = pd.DataFrame({'word': count_vectorizer_initial.get_feature_names(),
                         'count': pd.DataFrame(count_data_initial.toarray()).sum(axis=0).values})
vocab_df_sorted = vocab_df.sort_values(by=['count'], ascending=False)
# count data with idf values 
tf_vec = TfidfVectorizer()
count_data_idf = tf_vec.fit_transform(corpus)

for t in thresholds:
    t = round(t, 3)
    # document frequency based pruning
    count_vectorizer = CountVectorizer(min_df=t)
    count_data = count_vectorizer.fit_transform(corpus)
    pickle.dump(count_data, open(f"{path}/count_data_doc_freq_{_id}_{t}", 'wb'))
    pickle.dump(count_vectorizer.get_feature_names(), open(f"{path}/vocab_doc_freq_{_id}_{t}", 'wb'))
    # vocab length according to this threshold
    vocab_length = len(count_vectorizer.vocabulary_)

    # tf-idf based pruning
    indices = pd.DataFrame(count_data_idf.toarray()).mean(axis=0).sort_values()[::-1][0:vocab_length].index
    pickle.dump(count_data_initial[:, indices], open(f"{path}/count_data_idf_{_id}_{t}", 'wb'))
    pickle.dump([tf_vec.get_feature_names()[i] for i in indices], open(f"{path}/vocab_idf_{_id}_{t}", 'wb'))

    # term frequency based pruning
    # consider i top words in the vocabulary with highest frequency
    new_vocab_df = vocab_df_sorted[vocab_df_sorted['count'] >= vocab_df_sorted['count'].values[vocab_length - 1]]
    pickle.dump(count_data_initial[:,new_vocab_df.index], open(f"{path}/count_data_vocab_freq_{_id}_{t}", 'wb'))
    pickle.dump(new_vocab_df['word'].values, open(f"{path}/vocab_vocab_freq_{_id}_{t}", 'wb'))