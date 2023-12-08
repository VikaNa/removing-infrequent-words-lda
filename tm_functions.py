import numpy as np
import pandas as pd



def topic_word_df_sklearn(model, feature_names):
    """ make dataframe with word probas per topic from sklearn lda model and sklearn vectorizer feature names"""
    components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    jk = pd.DataFrame(components)
    jk.columns = feature_names
    jk = jk.T
    jk.columns = [str(i) for i in jk.columns.values]
    return jk

def get_topic_words_sklearn(model, count_vectorizer, n_top_words, frequency=False):
    import numpy as np
    all_topics = []
    words = count_vectorizer.get_feature_names()
    if frequency:
        for topic_idx, topic in enumerate(model.components_):
            frequent_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            values = np.sort(topic)[::-1][:n_top_words]
            topic_words = {word: value for word, value in zip(frequent_words,values)}
            all_topics.append(topic_words)
    else:
        for topic_idx, topic in enumerate(model.components_):
            # print("\nTopic #%d:" % topic_idx)
            topic_words = " ".join([words[i]
                                    for i in topic.argsort()[:-n_top_words - 1:-1]])
            all_topics.append(topic_words)
    return all_topics

def get_topic_words_sklearn_tf(model, words, n_top_words, frequency=False):
    import numpy as np
    all_topics = []
    if frequency:
        for topic_idx, topic in enumerate(model.components_):
            frequent_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            values = np.sort(topic)[::-1][:n_top_words]
            topic_words = {word: value for word, value in zip(frequent_words,values)}
            all_topics.append(topic_words)
    else:
        for topic_idx, topic in enumerate(model.components_):
            # print("\nTopic #%d:" % topic_idx)
            topic_words = " ".join([words[i]
                                    for i in topic.argsort()[:-n_top_words - 1:-1]])
            all_topics.append(topic_words)
    return all_topics

def topic_word_frequencies_matching(d1, d2, measure='cosine similarity', vocabulary = 'intersection', verbose = False):
    '''
    d1 and d2 are topic-word matrices to be compared.
    '''
    import numpy as np
    d2.columns = [str(i) + '_2' for i in range(d2.shape[1])]
    # join columns of another data frame, "how" default "left" (use calling frame's index, also indizes von der linken data frame)
    # andere Möglichkeiten für how right, outer (union of the indices), inner (intersection of the indices)
    # 'inner': form intersection of calling frame's index with other's index, preserving the order of the calling one.
    # d12 contains the words as row indexer, and topic names in the columns (the topics of the second dataframe are with the suffix _2)
    if vocabulary == 'intersection':
        d12 = d1.join(d2, how='inner')  # behalte nur die Wörter, die in beiden data frames vorkommen
    elif vocabulary == 'union':
        d12 = d1.join(d2, how='outer')
        d12 = d12.fillna(0)
    if verbose:
        print(f'Vocabulary size of the first model: {d1.shape[0]}')
        print(f'Vocabulary size of the second model: {d2.shape[0]}')
        print(f'Joint vocabulary size: {d12.shape[0]}')
    a = d12.filter(regex="^((?!_).)*$").columns.values.tolist() # column names of the first topic word dataframe
    b = d12.filter(like="_2").columns.values.tolist() # column names of the second topic word dataframe
    topic_assignment = []
    scores = []
    if measure == 'cosine similarity':
        from sklearn.metrics.pairwise import cosine_similarity
        for topic in a:
            similarities = [cosine_similarity([d12[topic].values], [d12[topic2].values])[0][0] for topic2 in b]
            assignment = np.argmax(similarities)  # extract the index (topic number) of the maximum similarity score
            # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
            score = similarities[assignment]
            topic_assignment.append(assignment)
            scores.append(score)
    elif measure == 'js distance':
        from scipy.spatial import distance
        for topic in a:
            similarities = [distance.jensenshannon(d12[topic].values.tolist(), d12[topic2].values.tolist()) for topic2 in b]
            assignment = np.argmin(similarities)  # extract the index (topic number) of the maximum similarity score
            # score = np.sort(similarities)[::-1][0] # extract the similarity score (maximum), np.sort does not work!!
            score = similarities[assignment]
            topic_assignment.append(assignment)
            scores.append(score)
    return topic_assignment, scores

def find_best_matching_topics(match):
    """
    Use the Hungarian Algorithm to find the cheapest matching of topics,
    where the cost measure is the Jensen-Shannan divergence or cosine distance between word distributions in topics.
    :param match: dataframe or np.array, topics in rows, word probs / counts in columns
    :param args: argument dictionary
    :return:
    """
    import re
    from scipy.optimize import linear_sum_assignment as lsa
    row_ind, col_ind = lsa(match)
    rownames = [int(match.index[index]) for index in row_ind]
    colnames = [[int(re.sub('_2', '', value)) for value in match.columns][index] for index in col_ind]
    # the topics of the first model as the row indices
    # the assigned topics of the second model in the first column
    assign = pd.DataFrame(data = colnames, index = rownames)
    return assign, row_ind, col_ind


def match_hungarian(d1, d2, method=['js', 'cosine'], merge = 'inner'):
    # d1,d2: rows = vocabulary words; columns = number of topics
    import itertools
    from scipy.spatial import distance
    # join columns of another data frame, "how" default "left" (use calling frame's index, also indizes von der linken data frame)
    # options for "how" are right, outer (union of the indices), inner (intersection of the indices)
    # 'inner': form intersection of calling frame's index with other's index, preserving the order of the calling one.
    d2.columns = [str(i) + '_2' for i in range(d2.shape[1])]
    ## choose
    d12 = d1.join(d2, how=merge)
    d12 = d12.fillna(0)
    # extract column names, a for the first data frame, b for the second one
    a = d12.filter(regex="^((?!_).)*$").columns.values.tolist()
    b = d12.filter(like="_2").columns.values.tolist()
    if method == 'js':
        sims = [[i, j, distance.jensenshannon(d12[i].values.tolist(), d12[j].values.tolist())] for i, j in itertools.product(a,b)]
        simsdf = pd.DataFrame(sims) #number of rows equals len(a)*len(b)
        match = simsdf.pivot(0, 1, 2)
        best_matching, row_ind, col_ind = find_best_matching_topics(match)
    elif method == 'cosine':
        sims = [[i, j, distance.cosine(d12[i].values.tolist(), d12[j].values.tolist())] for i, j in
                       itertools.product(a, b)]
        simsdf = pd.DataFrame(sims)
        match = simsdf.pivot(0, 1, 2)
        best_matching, row_ind, col_ind = find_best_matching_topics(match)

    return best_matching, match, row_ind, col_ind


def rbo_based_matching(topic_words_a, topic_words_b, p):
    import octis.evaluation_metrics.rbo as octis_rbo
    topic_assignment = []
    scores = []
    for topic in topic_words_a:
        similarities = [octis_rbo.rbo(topic.split(), topic2.split(), p)[2] for topic2 in topic_words_b]
        assignment = np.argmax(similarities)  # extract the index (topic number) of the maximum similarity score
        score = similarities[assignment]
        topic_assignment.append(assignment)
        scores.append(score)
    return topic_assignment, scores