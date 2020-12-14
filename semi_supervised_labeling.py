from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import torchtext
import data_module
import util
import numpy as np

from absl import app, flags

      
flags.DEFINE_boolean('debug', True, 'debug')

flags.DEFINE_float('portion_of_positive_samples', 0.5, 'portion_of_positive_samples')
flags.DEFINE_float('portion_of_labeled_training_data', 0.4, 'portion_of_labeled_training_data')

flags.DEFINE_integer('n_test_samples', 2000, 'n_test_samples')
flags.DEFINE_integer('n_train_samples', 5000, 'n_train_samples')

flags.DEFINE_integer('vocabulary_size', 20000, 'vocabulary_size')
flags.DEFINE_integer('word_emb_dims', 300, 'dimensionality of word embeddings (50, 100, 200 or 300)')

FLAGS = flags.FLAGS

def _main():
    FLAGS = flags.FLAGS

    data = data_module.get_data()
    train_data, test_data, droped_labels = data_module.get_train_test_split(data)

    if FLAGS.debug:
        # Test reproducibility
        #train_data[0] is the labeled training data
        train_data2, test_data2, droped_labels2 = data_module.get_train_test_split(data, verbose=False)
        assert(train_data[0].equals(train_data2[0]) and train_data[1].equals(train_data2[1]))
        assert(test_data.equals(test_data2))

 
    
    # get word embeddings
    embeddings = torchtext.vocab.GloVe(name="6B", dim=FLAGS.word_emb_dims, max_vectors=FLAGS.vocabulary_size)
    #Gets the embedding from Glove trained on the web (for max_vectors number of words)

    # train idf model from the labeled training set COULD ALSO BE TRAINED ON THE ENTIRE DATA..
    #vectorizer = util.get_tf_idf_vectorizer(train_data[0], embeddings.stoi)
    vectorizer = util.get_tf_idf_vectorizer(train_data[1], embeddings.stoi)

    # # TF-IDF feature_matrices
    train_X_l= util.get_tf_idf_review_embeddings(train_data[0], vectorizer)
    train_X_u= util.get_tf_idf_review_embeddings(train_data[1], vectorizer)
    test_X = util.get_tf_idf_review_embeddings(test_data, vectorizer)
    labeled_nodes = np.array(train_data[0]['label'])
    util.prediction(train_X_l, train_X_u, labeled_nodes, droped_labels)
    
    # Word Embedding feature matrices
    #train_X_l = util.get_distributed_review_embeddings(train_data[0], vectorizer, embeddings)
    #train_X_u = util.get_distributed_review_embeddings(train_data[1], vectorizer, embeddings)
    #test_X = util.get_distributed_review_embeddings(test_data, vectorizer, embeddings)

    w_lu = util.get_weight_matrix(train_X_l, train_X_u)
    w_ll = util.get_weight_matrix(train_X_l, train_X_l)
    w_ul = util.get_weight_matrix(train_X_u, train_X_l)
    
    w_uu = util.get_weight_matrix(train_X_u, train_X_u)
    d_uu = np.diag(np.sum(w_uu, axis=1))

    labeled_nodes = np.array(train_data[0]['label'])
    labs1 = util.labelize1(d_uu, w_uu, w_ul, labeled_nodes)
    print(labs1)
    
    labs2 = util.labelize2(d_uu, w_uu, w_ul, labeled_nodes)
    #print(labs2)
    print(np.array(droped_labels))
    
   # print(labs)    
    
    
   # util.visualize_top_k((test_X, train_X_l), (test_data, train_data[0]))

def main(args):
    for pld in [0.3]:
        FLAGS.portion_of_labeled_training_data = pld
        _main()
        print("")

if __name__ == '__main__':
    FLAGS.debug = True
    app.run(main)
