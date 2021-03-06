from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import torchtext
import data_module
import util
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy as sp


from absl import app, flags

      
flags.DEFINE_boolean('debug', True, 'debug')

flags.DEFINE_float('portion_of_positive_samples', 0.5, 'portion_of_positive_samples')
flags.DEFINE_float('portion_of_labeled_training_data', 1.0, 'portion_of_labeled_training_data')
flags.DEFINE_float('temperature', 0.03, 'temperature')

flags.DEFINE_integer('n_test_samples', 2000, 'n_test_samples')
flags.DEFINE_integer('n_train_samples', 5000, 'n_train_samples')

flags.DEFINE_integer('vocabulary_size', 20000, 'vocabulary_size')
flags.DEFINE_integer('word_emb_dims', 300, 'dimensionality of word embeddings (50, 100, 200 or 300)')
flags.DEFINE_string('emb_method', 'tfidf', 'method for generating review embeddings (tfidf or glove)')

flags.DEFINE_integer('random_seed', 111, 'random_seed')

flags.DEFINE_boolean('add_unlabeled_data', True, 'add_unlabeled_data')

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

def construct_W(labeled_train_X, unlabeled_train_X, test_X):
    if FLAGS.add_unlabeled_data and not ((unlabeled_train_X is None) or len(unlabeled_train_X) == 0):
        if type(unlabeled_train_X) == sp.sparse.csr_matrix:
            test_X = sp.sparse.vstack([unlabeled_train_X, test_X])
        else:
            test_X = np.concatenate([unlabeled_train_X, test_X], axis=0)

    w_lu = util.get_weight_matrix(labeled_train_X, test_X, t=FLAGS.temperature)
    w_ul = util.get_weight_matrix(test_X, labeled_train_X, t=FLAGS.temperature)
    w_ll = util.get_weight_matrix(labeled_train_X, labeled_train_X, t=FLAGS.temperature)
    w_uu = util.get_weight_matrix(test_X, test_X, t=FLAGS.temperature)
    W = np.block([[w_ll, w_lu], [w_ul, w_uu]])
    np.fill_diagonal(W, 0)

    return W


def train_and_evaluate_lp(train_data, test_data):
    labeled_train_X, unlabeled_train_X, test_X = get_embeddings(train_data, test_data)
    test_Y = test_data['label']

    W = construct_W(labeled_train_X, unlabeled_train_X, test_X)

    l = labeled_train_X.shape[0]
    t = test_X.shape[0]
    n = W.shape[0]

    fl = np.reshape(np.array(train_data[0]['label']), (-1, 1))
    L = np.diag(W.sum(axis=0)) - W

    # the harmonic function.
    Lu_factor = sp.linalg.cho_factor(L[l:, l:])
    fu = -sp.linalg.cho_solve(Lu_factor, L[l:, :l].dot(fl))

    fu[fu >= 1 / 2] = 1
    fu[fu < 1 / 2] = 0

    fu = np.reshape(fu, (-1))
    y_pred = fu[-t:]

    acc = round(accuracy_score(test_Y, y_pred) * 100, 4)

    return acc


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

    # Perform stratified K-Fold cross-validation
    val_accs = []
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(labeled_train_data, labeled_train_data['label']):
        temp_labeled_train_data = train_data[0].iloc[train_index]

        temp_train_data = (temp_labeled_train_data, unlabeled_train_data)
        temp_test_data = train_data[0].iloc[test_index]

        val_acc = train_and_evaluate_lp(temp_train_data, temp_test_data)
        val_accs.append(val_acc)

    val_acc = np.mean(val_accs)
    val_acc_std = round(np.std(val_accs), 4)
    val_acc = round(val_acc, 4)
    test_acc = train_and_evaluate_lp(train_data, test_data)

    print("Accuracy ({} features): val -> {} % (+- {}), test -> {} %".format('TF-IDF' if FLAGS.emb_method == 'tfidf' else 'GloVe', val_acc, val_acc_std, test_acc))

    return val_acc, test_acc


def main(args):
    FLAGS.emb_method = 'tfidf'

    plds = np.linspace(0.4, 1.0, 4)
    # plds = [1.0, 0.4]
    temps = np.logspace(-2, 0, 15)
    random_seeds = [111, 12345, 5, 135, 999]
    # random_seeds = [111, 12345]

    rows = 2
    cols = 2

    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(cols * 3.5 + (cols - 1) * 0.1,
                                                                           rows * 3 + (rows - 1) * 0.1)
                            )
    if cols == 1 and rows == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for idx, item in enumerate(zip(plds, axs)):
        pld, ax = item

        FLAGS.portion_of_labeled_training_data = pld
        val_accs_per_seed = []
        test_accs_per_seed = []

        for random_seed in random_seeds:
            FLAGS.random_seed = random_seed

            val_accs = []
            test_accs = []

            for temp in temps:
                FLAGS.temperature = temp
                val_acc, test_acc = _main()

                val_accs.append(val_acc)
                test_accs.append(test_acc)

            val_accs_per_seed.append(np.array(val_accs))
            test_accs_per_seed.append(np.array(test_accs))

        val_accs = np.stack(val_accs_per_seed, axis=0)
        val_accs_mean = np.mean(val_accs, axis=0)
        val_accs_std = np.std(val_accs, axis=0)

        test_accs = np.stack(test_accs_per_seed, axis=0)
        test_accs_mean = np.mean(test_accs, axis=0)
        test_accs_std = np.std(test_accs, axis=0)

        ax.set_xlabel('Temperature')
        ax.set_xscale('log')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{int(pld * 100)}% labeled data')

        if idx == (cols - 1):
            ax.errorbar(temps, val_accs_mean, val_accs_std, label='5-fold CV accuracy')
            ax.errorbar(temps, test_accs_mean, test_accs_std, label='Test set accuracy')
        else:
            ax.errorbar(temps, val_accs_mean, val_accs_std)
            ax.errorbar(temps, test_accs_mean, test_accs_std)

        ax.scatter(temps, val_accs_mean, color='black', marker='x')
        ax.scatter(temps, test_accs_mean, color='black', marker='x')

    handles, labels = axs[cols-1].get_legend_handles_labels()
    axs[cols-1].legend(handles, ['5-fold CV accuracy', 'Test set accuracy'])
    plt.show()

    if not FLAGS.debug:
        fig.savefig('semi_supervised_{}.png'.format('TFIDF' if FLAGS.emb_method == 'tfidf' else 'GloVe'))
        fig.savefig('semi_supervised_{}.pdf'.format('TFIDF' if FLAGS.emb_method == 'tfidf' else 'GloVe'))


if __name__ == '__main__':
    FLAGS.debug = True
    app.run(main)
