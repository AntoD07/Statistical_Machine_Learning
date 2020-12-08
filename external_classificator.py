from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torchtext
import data_module
import util

from absl import app, flags


flags.DEFINE_boolean('debug', True, 'debug')

flags.DEFINE_float('portion_of_positive_samples', 0.5, 'portion_of_positive_samples')
flags.DEFINE_float('portion_of_labeled_training_data', 1.0, 'portion_of_labeled_training_data')

flags.DEFINE_integer('n_test_samples', 2000, 'n_test_samples')
flags.DEFINE_integer('n_train_samples', 5000, 'n_train_samples')

flags.DEFINE_integer('vocabulary_size', 20000, 'vocabulary_size')
flags.DEFINE_integer('word_emb_dims', 300, 'dimensionality of word embeddings (50, 100, 200 or 300)')

FLAGS = flags.FLAGS


def train_and_evaluate_random_forest_classifier(train_X, train_Y, test_X, test_Y):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(train_X, train_Y)
    y_pred_rf = clf_rf.predict(test_X)

    return round(accuracy_score(test_Y, y_pred_rf) * 100, 4)

def _main():
    FLAGS = flags.FLAGS

    data = data_module.get_data()
    train_data, test_data = data_module.get_train_test_split(data)

    if FLAGS.debug:
        # Test reproducibility
        train_data2, test_data2 = data_module.get_train_test_split(data, verbose=False)
        assert(train_data[0].equals(train_data2[0]) and train_data[1].equals(train_data2[1]))
        assert(test_data.equals(test_data2))

    # get word embeddings
    embeddings = torchtext.vocab.GloVe(name="6B", dim=FLAGS.word_emb_dims, max_vectors=FLAGS.vocabulary_size)

    # learn idf from the training set
    vectorizer = util.get_tf_idf_vectorizer(train_data[0], embeddings.stoi)

    #####
    train_Y = train_data[0]['label']
    test_Y = test_data['label']

    # Word Embeddings + RandomForest
    train_X = util.get_distributed_review_embeddings(train_data[0], vectorizer, embeddings)
    test_X = util.get_distributed_review_embeddings(test_data, vectorizer, embeddings)
    acc_word_embeddings = train_and_evaluate_random_forest_classifier(train_X, train_Y, test_X, test_Y)
    print("Accuracy (word embeddings): {} %".format(acc_word_embeddings))

    # TF-IDF + RandomForest
    train_X = util.get_tf_idf_review_embeddings(train_data[0], vectorizer)
    test_X = util.get_tf_idf_review_embeddings(test_data, vectorizer)
    acc_word_embeddings = train_and_evaluate_random_forest_classifier(train_X, train_Y, test_X, test_Y)
    print("Accuracy (TF-IDF features): {} %".format(acc_word_embeddings))

def main(args):
    for pld in [0.5, 1]:
        FLAGS.portion_of_labeled_training_data = pld
        _main()
        print("")

if __name__ == '__main__':
    FLAGS.debug = False
    app.run(main)


