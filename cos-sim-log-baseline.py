from tools import df_split, cos_sim
import random
import statsmodels.api as sm
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # set random seed for consistency, load data
    random.seed(0)
    df_q = pd.read_csv('data/quora_duplicate_questions.tsv', sep="\t")
    q1_split, q1_full = df_split(df_q, 'question1')
    q2_split, q2_full = df_split(df_q, 'question2')
    # get cos sim, get difference in length between the two questions (both features in our model)
    cos = cos_sim(q1_split, q2_split)
    len_diff = (q1_full.str.len() - q2_full.str.len()).values
    # prepare features
    x = np.vstack([cos, len_diff]).T
    y = df_q['is_duplicate'].values
    x = sm.add_constant(x)
    # randomly generate n / 2 indices. we then use these indices to create a train/test set (50/50 split here)
    n = y.shape[0]
    train_ind = random.sample(range(0, n), int(n / 2))
    test_ind = list(set(range(0, n)) - set(train_ind))

    xtrain, xtest = x[train_ind], x[test_ind]
    ytrain, ytest = y[train_ind], y[test_ind]
    # train logistic regression, find accuracy on test set
    model = sm.Logit(ytrain, xtrain).fit()
    ypred = (model.predict(xtest) > 0.5).astype(int)
    accuracy = (ypred == ytest).sum() / len(ypred)
    print(accuracy)
