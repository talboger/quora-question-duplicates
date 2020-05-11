import numpy as np
import string
from nltk.corpus import stopwords
import torchtext.data as tt


def remove_punctuation(q):
    # remove punctuation of string (important for consistency in cos sim function)
    # no params; only used in df.apply(remove_punctuation)
    # returns desired section of df without punctuation
    q = ''.join([i for i in q if i not in frozenset(string.punctuation)])
    return q


def train_valid_test_split(df, train_perc=.5, valid_perc=.25, seed=0):
    # this is not used in any other file. this is how i separated the main file into train/valid/test files
    # to make the separation for yourself, load the df then call train_valid_test_split(df)
    # param: df: the dataframe to split
    # params: train_perc, valid_perc: percent of data to assign to train and validation splits
    # param: seed (set at 0 just for consistent replication; only determines the numpy random
    # no return values; saves train, valid, test csv

    df_clean = df.copy()
    q1_split, q1_full = df_split(df_clean, 'question1')
    q2_split, q2_full = df_split(df_clean, 'question2')
    # this limits our question length to 50. it's important for simplicity in the torchtext field later on
    drop_list = []
    for i in range(len(q1_split)):
        if len(q1_split[i]) > 50:
            drop_list.append(i)
        elif len(q2_split[i]) > 50:
            drop_list.append(i)
    df_clean = df_clean.drop(drop_list).reset_index(drop=True)
    drop_num, total = len(drop_list), len(df)
    print(f"Dropped {drop_num:.0f} rows out of {total:.0f} total")  # dropped only 906/404290 rows, or 0.22%

    np.random.seed(seed)
    perm = np.random.permutation(df_clean.index)
    n = len(df_clean.index)
    df_clean[['question1', 'question2']] = df_clean[['question1', 'question2']].astype(str)
    df_clean['question1'] = df_clean['question1'].apply(remove_punctuation)
    df_clean['question2'] = df_clean['question2'].apply(remove_punctuation)
    train_end = int(train_perc * n)
    valid_end = int(valid_perc * n) + train_end
    train = df_clean.iloc[perm[:train_end]]
    valid = df_clean.iloc[perm[train_end:valid_end]]
    test = df_clean.iloc[perm[valid_end:]]
    for i, j in zip([train, valid, test], ["train", "valid", "test"]):
        i.reset_index(drop=True).to_csv('data/' + j + '.csv', index=False)


def df_split(df, col):
    # clean column of sentences in df into list of tokenized words
    # param: df, col: desired dataframe and column to clean
    # return: tuple: first return value: list of the tokenized column (lowercase and w/o punctuation)
    # return: tuple: second return value: pd.Series of the non-tokenized but cleaned sentences (used for len_diff)
    df[col] = df[col].astype(str)
    no_punc = df[col].apply(remove_punctuation)
    return [word.lower().split() for word in no_punc.tolist()], no_punc


def cos_sim(l1, l2):
    # calculate cosine similarity between two lists of words
    # param: l1, l2: 2 lists of tokens
    # return: cos_list, a list giving the pairwise cosine similarity between each sentence in l1 and l2

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


def prep_torch_data(batch_size):
    # creates torchtext fields and iterators of train, valid, and test csvs
    # param: batch_size: desired batch_size of bucket iterator
    # return: iterators, torchtext fields
    tokenize = lambda x: x.split()
    TEXT = tt.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=50)
    LABEL = tt.Field(sequential=False, use_vocab=False)

    fields = [("id", None), ("qid1", None), ("qid2", None),
              ("question1", TEXT), ("question2", TEXT),
              ("is_duplicate", LABEL)]

    train, valid = tt.TabularDataset.splits(path="data",
                                            train="train.csv", validation="valid.csv",
                                            format="csv", skip_header=True,
                                            fields=fields)
    test = tt.TabularDataset.splits(path="data/test.csv",
                                    format="csv", skip_header=True,
                                    fields=fields)

    TEXT.build_vocab(train, valid, test)
    train_iter, val_iter, test_iter = tt.BucketIterator.splits((train, valid, test),
                                                               batch_size=batch_size,
                                                               sort_key=lambda x: len(x.question1))

    return train_iter, val_iter, test_iter, TEXT, LABEL
