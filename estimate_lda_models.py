import numpy as np
import os
import pickle
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Define thresholds for preprocessing
thresholds = np.arange(0.0, 0.1, 0.005)
k_true = 50
# For each corpus, estimate LDA models varying the min_df threshold

# Corpus ID
_id = 1
# Corpus path
path = 'results/corpora'
starttime = time.time()
corpus = pd.read_pickle(open(f"{path}/corpus_{_id}.pkl", 'rb'))
for t in thresholds:
    t = round(t, 3)
    ## choose the technique and comment out other techniques
    count_data = pickle.load(open(f"{path}/count_data_doc_freq_{_id}_{t}", 'rb')) # relative document frequency
    #count_data = pickle.load(open(f"{path}/count_data_idf_{_id}_{t}", 'rb')) # tf-idf value
    #count_data = pickle.load(open(f"{path}/count_data_vocab_freq_{_id}_{t}", 'rb')) # absolute term frequency
    lda = LatentDirichletAllocation(
        n_components=k_true,
        max_iter=100,
        random_state=0).fit(count_data)
    # save the models
    models_path = f'results/models/corpus_{_id}'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    pickle.dump(lda, open(f'{models_path}/lda_mindf_{t}', 'wb'))

print('That took {} seconds'.format(time.time() - starttime))

