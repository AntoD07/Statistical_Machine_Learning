from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torchtext
import data_module
import util
import numpy as np
import matplotlib.pyplot as plt

from absl import app, flags


flags.DEFINE_boolean('debug', True, 'debug')

flags.DEFINE_float('portion_of_positive_samples', 0.5, 'portion_of_positive_samples')
flags.DEFINE_float('portion_of_labeled_training_data', 1.0, 'portion_of_labeled_training_data')

flags.DEFINE_integer('n_test_samples', 2000, 'n_test_samples')
flags.DEFINE_integer('n_train_samples', 5000, 'n_train_samples')

flags.DEFINE_integer('vocabulary_size', 20000, 'vocabulary_size')
flags.DEFINE_integer('word_emb_dims', 300, 'dimensionality of word embeddings (50, 100, 200 or 300)')
flags.DEFINE_string('emb_method', 'glove', 'method for generating review embeddings (tfidf or glove)')

flags.DEFINE_integer('random_seed', 111, 'random_seed')

FLAGS = flags.FLAGS

def get_embeddings(train_data, test_data):
    labeled_train_data, unlabeled_train_data = train_data

    # get word embeddings
    embeddings = torchtext.vocab.GloVe(name="6B", dim=FLAGS.word_emb_dims, max_vectors=FLAGS.vocabulary_size)

    # train idf model from the labeled training set
    vectorizer = util.get_tf_idf_vectorizer(train_data, embeddings.stoi)

    if FLAGS.emb_method == 'tfidf':
        # TF-IDF feature_matrices
        labeled_train_X = util.get_tf_idf_review_embeddings(labeled_train_data, vectorizer)
        unlabeled_train_X = util.get_tf_idf_review_embeddings(unlabeled_train_data, vectorizer)
        test_X = util.get_tf_idf_review_embeddings(test_data, vectorizer)
    elif FLAGS.emb_method == 'glove':
        # Word Embedding feature matrices
        labeled_train_X = util.get_distributed_review_embeddings(labeled_train_data, vectorizer, embeddings)
        unlabeled_train_X = util.get_distributed_review_embeddings(unlabeled_train_data, vectorizer, embeddings)
        test_X = util.get_distributed_review_embeddings(test_data, vectorizer, embeddings)

    return labeled_train_X, unlabeled_train_X, test_X

def train_random_forest_classifier(train_X, train_Y):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(train_X, train_Y)

    return clf_rf


def train_and_evaluate_random_forest_classifier(train_X, train_Y, test_X, test_Y):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(train_X, train_Y)
    y_pred_rf = clf_rf.predict(test_X)

    return round(accuracy_score(test_Y, y_pred_rf) * 100, 4)

def _main():
    FLAGS = flags.FLAGS
    assert FLAGS.emb_method in {'tfidf', 'glove'}

    data = data_module.get_data()
    train_data, test_data, _ = data_module.get_train_test_split(data, random_seed=FLAGS.random_seed)
    labeled_train_data, unlabeled_train_data = train_data

    if FLAGS.debug:
        # Test reproducibility
        train_data2, test_data2, droped_labels2 = data_module.get_train_test_split(data, verbose=False, random_seed=FLAGS.random_seed)
        assert(train_data[0].equals(train_data2[0]) and train_data[1].equals(train_data2[1]))
        assert(test_data.equals(test_data2))

    labeled_train_X, unlabeled_train_X, test_X = get_embeddings(train_data, test_data)
    labeled_train_Y = labeled_train_data['label']
    test_Y = test_data['label']

    # Perform stratified K-Fold cross-validation
    val_accs = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(labeled_train_X, labeled_train_Y):
        temp_train_X = labeled_train_X[train_index]
        temp_train_Y = labeled_train_Y[train_index]

        temp_test_X = labeled_train_X[test_index]
        temp_test_Y = labeled_train_Y[test_index]

        val_acc = train_and_evaluate_random_forest_classifier(temp_train_X, temp_train_Y, temp_test_X, temp_test_Y)
        val_accs.append(val_acc)

    val_acc = np.mean(val_accs)
    val_acc_std = round(np.std(val_accs), 4)
    val_acc = round(val_acc, 4)
    test_acc = train_and_evaluate_random_forest_classifier(labeled_train_X, labeled_train_Y, test_X, test_Y)
    train_acc = train_and_evaluate_random_forest_classifier(labeled_train_X, labeled_train_Y, labeled_train_X, labeled_train_Y)

    print("Accuracy ({} features): val -> {} % (+- {}), test -> {} %".format('TF-IDF' if FLAGS.emb_method == 'tfidf' else 'GloVe', val_acc, val_acc_std, test_acc))

    return train_acc, val_acc, test_acc

def main(args):
    plds = np.linspace(0.4, 1.0, 4)
    # plds = [1.0]

    random_seeds = [111, 12345, 5, 135, 999]

    train_accs_mean = []
    train_accs_std = []

    val_accs_mean = []
    val_accs_std = []

    test_accs_mean = []
    test_accs_std = []

    for pld in plds:
        FLAGS.portion_of_labeled_training_data = pld

        train_accs = []
        val_accs = []
        test_accs = []

        for random_seed in random_seeds:
            FLAGS.random_seed = random_seed
            train_acc, val_acc, test_acc = _main()
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

        train_accs_mean.append(np.mean(train_accs))
        val_accs_mean.append(np.mean(val_accs))
        test_accs_mean.append(np.mean(test_accs))

        train_accs_std.append(np.std(train_accs))
        val_accs_std.append(np.std(val_accs))
        test_accs_std.append(np.std(test_accs))

    rows = 1
    cols = 1

    fig, ax = plt.subplots(rows, cols, sharey=True, figsize=(cols * 3.5 + (cols - 1) * 0.1,
                                                             rows * 3 + (rows - 1) * 0.1))

    ax.errorbar(plds, train_accs_mean, train_accs_std, label='Train accuracy')
    ax.errorbar(plds, val_accs_mean, val_accs_std, label='5-fold CV accuracy')
    ax.errorbar(plds, test_accs_mean, test_accs_std, label='Test accuracy')

    ax.scatter(plds, train_accs_mean, color='black', marker='x')
    ax.scatter(plds, val_accs_mean, color='black', marker='x')
    ax.scatter(plds, test_accs_mean, color='black', marker='x')

    plt.legend()
    plt.show()

    if not FLAGS.debug:
        fig.savefig('external_classifier_{}.png'.format('TFIDF' if FLAGS.emb_method == 'tfidf' else 'GloVe'))
        fig.savefig('external_classifier_{}.pdf'.format('TFIDF' if FLAGS.emb_method == 'tfidf' else 'GloVe'))

if __name__ == '__main__':
    FLAGS.debug = False
    app.run(main)


