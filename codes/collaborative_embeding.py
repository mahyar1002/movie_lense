from sklearn.decomposition import TruncatedSVD
import pandas as pd
from helper import log, save_embeddings, save_data
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import ipdb
from authoencoder import AutoEncoder


class CollaborativeFiltering(object):
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.ids = None
        # self.movies = ratings_df.index.tolist()
        # self.users = ratings_df.columns.tolist()

    def create_rating_matrix(self):
        # convert the ratings data file to a pandas dataframe
        log('Info', "###### Pivoting Ratings ######")
        ratings_df_matrix = self.ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.ids = ratings_df_matrix.index.tolist()
        log('Info', "shape: {}".format(ratings_df_matrix.shape))
        log('Info', '\n{}'.format(ratings_df_matrix.head()))
        log('Info', "###### End of Pivoting Ratings ######")

        return ratings_df_matrix

    def create_normalized_numpy_ratings(self):
        ratings_df_matrix = self.create_rating_matrix()
        log('Info', "###### Normalizing Ratings ######")
        R = ratings_df_matrix.values
        user_ratings_mean = np.mean(R, axis=1)
        rating_demeaned = R - user_ratings_mean.reshape(-1, 1)
        log('Info', "###### End of Normalizing Ratings ######")

        return rating_demeaned

    def get_svd_embeddings(self, feature_matrix, n):
        log('Info', "###### Start SVD on Ratings ######")
        svd = TruncatedSVD(n_components=n)
        latent_matrix = svd.fit_transform(feature_matrix)
        latent_df = pd.DataFrame(latent_matrix, index=self.ids)
        log('Info', "###### End of SVD on Ratings ######")
        return latent_df

    def get_authoencoder_embedings(self, feature_df):
        ae = AutoEncoder(feature_df, validation_perc=0.1, lr=1e-3, intermediate_size=1000, encoded_size=100,
                         cuda_enable=False)
        ae.train_loop(epochs=15)
        losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
        losses['epoch'] = (losses.index + 1) / 3
        encoded = ae.get_encoded_representations()
        return losses, encoded

    def apply(self, method='svd'):
        feature_matrix = self.create_normalized_numpy_ratings()
        save_data(feature_matrix, 'result/rating_demeaned.pkl')

        if method == 'svd':
            latent_df = self.get_svd_embeddings(feature_matrix, 100)
            log("Info", "shape: {}".format(latent_df.shape))
            log("Info", "\n{}".format(latent_df.head(5)))
        elif method == 'ae':
            feature_df = pd.DataFrame(feature_matrix)
            losses, latent_np = self.get_authoencoder_embedings(feature_df)
            latent_df = pd.DataFrame(latent_np, index=self.ids)
            log("Info", "shape: {}".format(latent_df.shape))
            save_embeddings(losses, 'result/collaborative_losses.pkl', 'pickle')
        else:
            raise AssertionError("Please specify a correct method [svd, ae]")

        save_data(latent_df, 'result/collaborative_embeddings_{}.pkl'.format(method))
