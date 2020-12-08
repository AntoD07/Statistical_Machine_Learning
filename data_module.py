from absl import flags

import random

import pandas as pd
import numpy as np

FLAGS = flags.FLAGS

def test():
    if FLAGS.debug:
        print("DEBUG")
    else:
        print("NOT DEBUG")

def get_n_train_samples():
    if FLAGS.debug:
        return 20

    return FLAGS.n_train_samples

def get_n_test_samples():
    if FLAGS.debug:
        return 10

    return FLAGS.n_test_samples

def get_data():
    data = pd.read_csv("IMDB Dataset.csv")
    data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data['label'] = data['label'].astype('int')

    return data

def get_train_test_split(data, verbose=True):
    n_train_samples = get_n_train_samples()
    n_test_samples = get_n_test_samples()

    if verbose:
        print("Number of train samples:", n_train_samples)
        print("Number of test samples:", n_test_samples)

    portion_of_positive_samples = FLAGS.portion_of_positive_samples
    portion_of_labeled_training_data = FLAGS.portion_of_labeled_training_data

    pos_data = data[data['label'] == 1]
    neg_data = data[data['label'] == 0]

    n_pos_test = int(n_test_samples * portion_of_positive_samples)
    n_neg_test = n_test_samples - n_pos_test

    n_pos_train = int(n_train_samples * portion_of_positive_samples)
    n_neg_train = n_train_samples - n_pos_train

    n_labeled_train = int(n_train_samples * portion_of_labeled_training_data)
    if verbose:
        print("Number of labeled training samples:", n_labeled_train)

    random.seed(123)
    np.random.seed(123)

    # === Sample test data ===
    pos_idxs = random.sample(range(len(pos_data)), n_pos_test)
    neg_idxs = random.sample(range(len(neg_data)), n_neg_test)

    test_data = pd.concat([pos_data.iloc[pos_idxs], neg_data.iloc[neg_idxs]], axis=0)
    test_data = test_data.sample(frac=1).reset_index(drop=True)  # shuffle positive and negative reviews

    # Remove test data from data
    pos_data = pos_data.iloc[list(set(range(len(pos_data))) - set(pos_idxs))]
    neg_data = neg_data.iloc[list(set(range(len(neg_data))) - set(neg_idxs))]

    # === Sample train data ===
    pos_idxs = random.sample(range(len(pos_data)), n_pos_train)
    neg_idxs = random.sample(range(len(neg_data)), n_neg_train)

    train_data = pd.concat([pos_data.iloc[pos_idxs], neg_data.iloc[neg_idxs]], axis=0)
    train_data = train_data.sample(frac=1).reset_index(drop=True)  # shuffle positive and negative reviews

    train_data_labeled = train_data[:n_labeled_train]
    train_data_unlabeled = train_data[n_labeled_train:]
    train_data_unlabeled.drop(['sentiment', 'label'], axis=1)

    return (train_data_labeled, train_data_unlabeled), test_data
