import numpy as np
import string
from nltk.corpus import stopwords


def remove_punctuation(q):
    # remove punctuation of string (important for consistency in cos sim function) - used in .apply
    q = ''.join([i for i in q if i not in frozenset(string.punctuation)])
    return q


def df_split(df, col):
    # clean column of sentences in df into list of tokenized words
    df[col] = df[col].astype(str)
    no_punc = df[col].apply(remove_punctuation)
    return [word.lower().split() for word in no_punc.tolist()], no_punc


def cos_sim(l1, l2):
    # calculate cosine similarity between two lists of words

    # define stop words, or words that aren't relevant to the "meaning" of a question
    # examples of stopwords include "the", "a", etc.
    sw = stopwords.words('english')

    l1_tokens, l2_tokens = [], []
    for i in range(len(l1)):
        # extract non-stopwords tokens from each list
        l1_set = {word for word in l1[i] if not word in sw}
        l2_set = {word for word in l2[i] if not word in sw}
        l1_ind, l2_ind = [], []
        # define union of the two lists, and assign 1 if the token is share between the two, 0 o/w
        union = l1_set.union(l2_set)
        for word in union:
            if word in l1_set:
                l1_ind.append(1)
            else:
                l1_ind.append(0)
            if word in l2_set:
                l2_ind.append(1)
            else:
                l2_ind.append(0)
        l1_tokens.append(l1_ind), l2_tokens.append(l2_ind)
    # calculate cosine similarity
    cos_list = []
    for i in range(len(l1)):
        cos = np.dot(l1_tokens[i], l2_tokens[i]) / (np.linalg.norm(l1_tokens[i]) * np.linalg.norm(l2_tokens[i]))
        cos_list.append(cos if not np.isnan(cos) else 0)  # adjusts for vectors of norm 0
    return cos_list
