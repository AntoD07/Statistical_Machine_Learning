import numpy as np
import torch
import scipy
import util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def get_weight_matrix(X1, X2, t=0.03):
    """T is a temperature parameter, which is fixed to 0.03 in the paper
    (but actually should be a hyper-parameter)"""
    cosine_sim = cosine_similarity(X1, X2, True)

    if FLAGS.debug and type(X1) == scipy.sparse.csr_matrix:
        cosine_sim2 = cosine_similarity(X1.todense(), X2.todense(), True)
        assert np.allclose(cosine_sim, cosine_sim2)

    dist = - (1 - cosine_sim) * 1/t

    # Prevent over/underflow
    dist[dist < -25] = -25
    dist[dist > 25] = 25

    dist = np.exp(dist)
    return dist

def visualize_top_k(embs, data, idx=0, k=3):
    X1, X2 = embs
    data1, data2 = data

    w = util.get_weight_matrix(X1, X2)
    distances = w[idx]

    top_k = np.argsort(distances, -1)[-k:][::-1]

    print(data1.iloc[idx]['review'])
    for i in top_k:
        print(data2.iloc[i]['review'])