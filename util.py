import numpy as np
import torch
import scipy
#import util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.gaussian_process.kernels import RBF
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from absl import flags

FLAGS = flags.FLAGS


def get_distributed_review_embeddings(data, vectorizer, embeddings):
    tf_idf_rep = get_tf_idf_review_embeddings(data, vectorizer)

    review_embs = []
    for row in tf_idf_rep:
        row = torch.tensor(np.array(row.todense())).type(torch.float32)
        review_emb = torch.mm(row, embeddings.vectors)
        review_embs.append(review_emb[0])

    review_embs = torch.stack(review_embs)
    return review_embs.numpy()


def get_tf_idf_review_embeddings(data, vectorizer):
    """Note that this returns a sparse matrix in the csr format"""
    tf_idf_rep = vectorizer.transform(data['review'])

    return tf_idf_rep

def get_tf_idf_vectorizer(data, vocabulary):
    """Data used to learn the idf. We could use the labeled training data only,
    or use the unlabeled training data as well"""
    vectorizer = TfidfVectorizer(use_idf=True,
                                 lowercase=True,
                                 strip_accents='ascii',
                                 norm='l2',
                                 vocabulary=vocabulary)

    vectorizer.fit(data['review'])

    return vectorizer

def get_weight_matrix(X1, X2, t= 0.03):
    """T is a temperature parameter, which is fixed to 0.03 in the paper
    (but actually should be a hyper-parameter)"""
    cosine_sim = cosine_similarity(X1, X2, True)

    if FLAGS.debug and type(X1) == scipy.sparse.csr_matrix:
        cosine_sim2 = cosine_similarity(X1.todense(), X2.todense(), False)
        assert np.allclose(cosine_sim, cosine_sim2)

    dist = - (1 - cosine_sim) * 1/t
   #dist = (1 + cosine_sim) * 1/2

    # Prevent over/underflow
    dist[dist < -25] = -25
    dist[dist > 25] = 25

    dist = np.exp(dist)
    return dist



def labelize1(D_uu, W_uu, W_ul, labeled_nodes):
    epsilon = 5e-4
    #Version 1 :
    Lapl = D_uu - W_uu
    Lapl = epsilon*np.ones(np.shape(Lapl))+ (1-epsilon)*Lapl
    Green = np.linalg.inv(Lapl)
    
    return Green @ W_ul @ labeled_nodes
    
def labelize2(D_uu, W_uu, W_ul, labeled_nodes):    
    #Version 2 computing matrix P
    
    p_uu = np.linalg.inv(D_uu)@ W_uu
    p_ul = np.linalg.inv(D_uu)@ W_ul
    mat = np.eye(len(p_uu))- p_uu
    #print("Size of (I - Puu) matrix : ", np.shape(mat))
    #print("Rank of (I - Puu) matrix : ", np.linalg.matrix_rank(mat))
    mat_inv = np.linalg.inv(mat)
   # print(mat_inv)
    
    return mat_inv@p_ul@labeled_nodes
def accuracy(prediction, true_value, number_labeled):
    prediction = prediction[number_labeled:]
    true_value = true_value[number_labeled:]
    N = len(true_value)
    print("Number of nodes to predict :", N)
    assert(len(prediction)==len(true_value))
    counter = 0
    return N**-1*np.sum([prediction[i] == true_value[i] for i in range(N)])
    

def prediction(X_l, X_u, labeled_out, groundtruth):
    n_u = X_u.shape[0]
    print("number of unkown nodes : ", X_u.shape[0])
    n_l = X_l.shape[0]
    print("number of known nodes : ", X_l.shape[0])
    X_l = X_l.toarray()
    X_u = X_u.toarray()
   # X = scipy.sparse.vstack((X_l, X_u), format='csr')
    X = np.concatenate((X_l,X_u)) 
    scale_vector = np.var(X, axis=0)
    #print(X.shape[0])
    print("Shape of X data regrouping labeled and unlabeled: ", np.shape(X))
    a = np.full(n_u, -1)
    y = np.concatenate((np.array(labeled_out), a), axis = 0)
    label_spread = LabelSpreading(kernel='rbf', alpha=0.1)
    label_spread.fit(X, y)
    output_labels = label_spread.transduction_
    true_labels = np.concatenate((labeled_out, groundtruth))
    print("Accuracy = ", accuracy(output_labels, true_labels, n_l))
    
  
    

def visualize_top_k(embs, data, idx=0, k=3):
    X1, X2 = embs
    data1, data2 = data

    w = get_weight_matrix(X1, X2)
    distances = w[idx]

    top_k = np.argsort(distances, -1)[-k:][::-1]

    print(data1.iloc[idx]['review'])
    for i in top_k:
        print(data2.iloc[i]['review'])