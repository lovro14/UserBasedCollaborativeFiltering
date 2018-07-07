import numpy as np
from numpy import newaxis


class RecommenderSystem(object):

    def __init__(self):
        pass

    def _calculate_similarity(self, ratings, epsilon=1e-9):
        sim = ratings.dot(ratings.T) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    def _normalize_data(self, ratings, mean):
        matrix_norm = np.zeros((ratings.shape[0], ratings.shape[1]))
        for i in range(ratings.shape[0]):
            non_zero_idx = ratings[i] != 0
            matrix_norm[i, non_zero_idx] = ratings[i, non_zero_idx] - mean[i]
        return matrix_norm

    def _calculate_mean(self, ratings):
        return np.true_divide(ratings.sum(1), (ratings != 0).sum(1))

    def predict_topk_nobias(self, ratings, k=40):
        pred = np.zeros(ratings.shape)
        mean = self._calculate_mean(ratings)
        ratings = self._normalize_data(ratings, mean)
        similarity = self._calculate_similarity(ratings)
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        mean_reshaped = mean[:, newaxis]
        pred += mean_reshaped
        return pred

