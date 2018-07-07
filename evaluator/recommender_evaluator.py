import numpy as np
from collections import defaultdict


class RecommenderEvaluator(object):

    def __init__(self):
        pass

    def rmse(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return np.sqrt(np.mean(np.power(pred - actual, 2)))

    def mae(self, pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return np.mean(np.abs(pred - actual))

    def precision_recall_at_k(self, predictions, test, mean_test, user_number, k=20):

        rated_movies = test > 0.0

        user_est_true = defaultdict(list)

        for userId in range(0, user_number):
            real_ratings = test[userId][rated_movies[userId]]
            predicted_ratings = predictions[userId][rated_movies[userId]]
            for real, predicted in zip(real_ratings, predicted_ratings):
                user_est_true[userId].append((predicted, real))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            if len(user_ratings) >= k:
                user_ratings.sort(key=lambda x: x[0], reverse=True)
                n_rel = sum((true_r >= mean_test[uid]) for (_, true_r) in user_ratings)
                n_rec_k = sum((est >= mean_test[uid]) for (est, _) in user_ratings[:k])
                n_rel_and_rec_k = sum(((true_r >= mean_test[uid]) and (est >= mean_test[uid]))
                                      for (est, true_r) in user_ratings[:k])
                precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
                recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

    def f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall)