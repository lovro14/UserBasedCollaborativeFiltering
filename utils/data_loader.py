import pandas as pd
import numpy as np
import os
from sklearn import cross_validation as cv


class DataLoader(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def load_data(self):
        if self.dataset == 'ml-100k':
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-100k/u.data'),
                sep='\t',
                engine="python",
                encoding="latin-1",
                names=['userid', 'itemid', 'rating', 'timestamp'])
        elif self.dataset == 'ml-1m':
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-1m/ratings.dat'),
                sep='::',
                engine="python",
                encoding="latin-1",
                names=['userid', 'itemid', 'rating', 'timestamp'])
        else:
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-20m/ratings.csv'),
                sep=',',
                header=None,
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python')
        self._prepare_dataset()

    def _prepare_dataset(self):
        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m':
            for column in self.rating_data:
                if column == "userid" or column == "itemid":
                    self.rating_data[column] = self.rating_data[column].astype(np.int32)
                if column == "rating":
                    self.rating_data[column] = self.rating_data[column].astype(np.float32)
            self.ratings = np.zeros((self.rating_data.userid.max(), self.rating_data.itemid.max()))
        else:
            self.rating_data.drop(self.rating_data.index[[0]], inplace=True)
            for column in self.rating_data:
                if column == "userId" or column == "movieId":
                    self.rating_data[column] = self.rating_data[column].astype(np.int32)
                if column == "rating":
                    self.rating_data[column] = self.rating_data[column].astype(np.float32)
            self.ratings = np.zeros((self.rating_data.userId.max(), self.rating_data.movieId.max()))

        for row in self.rating_data.itertuples():
            self.ratings[row[1] - 1, row[2] - 1] = row[3]

    def train_test_split(self):
        train_data_cv, test_data_cv = cv.train_test_split(self.rating_data, test_size=0.20)

        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m':
            train = np.zeros((self.rating_data.userid.max(), self.rating_data.itemid.max()))
            test = np.zeros((self.rating_data.userid.max(), self.rating_data.itemid.max()))
        else:
            train = np.zeros((self.rating_data.userId.max(), self.rating_data.movieId.max()))
            test = np.zeros((self.rating_data.userId.max(), self.rating_data.movieId.max()))

        for line in train_data_cv.itertuples():
            train[line[1] - 1, line[2] - 1] = line[3]

        for line in test_data_cv.itertuples():
            test[line[1] - 1, line[2] - 1] = line[3]
        return train, test


    def get_dataset_info(self):
        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m':
            a = user_number = self.rating_data['userid'].unique()

            user_number = self.rating_data['userid'].unique().shape[0]
            item_number = self.rating_data['itemid'].unique().shape[0]
        else:
            user_number = self.rating_data['userId'].unique().shape[0]
            item_number = self.rating_data['movieId'].unique().shape[0]
        return user_number, item_number


    def get_interatction_matrix_sparsity(self):
        sparsity = float(len(self.rating_data.nonzero()[0]))
        sparsity /= (self.rating_data.shape[0] * self.rating_data.shape[1])
        sparsity *= 100
        #print('Sparsity: {:4.2f}%'.format(sparsity))
        return sparsity
